#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/gdn_primitives.h"
#include "metal/kernels/common/gguf_sgmatrix.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"
constant uint kDecodeSimdgroups=8;
template <uint HeadDim> struct GdnDecodeShared {
 float projectK[8][32],projectQ[8][32];
 bfloat queries[JUSTFIT_TARGET_VERIFY_ROWS*HeadDim];
 bfloat keys[JUSTFIT_TARGET_VERIFY_ROWS*HeadDim];
 bfloat values[JUSTFIT_TARGET_VERIFY_ROWS*HeadDim];
 bfloat rows[JUSTFIT_TARGET_VERIFY_ROWS*HeadDim];
 float decay[JUSTFIT_TARGET_VERIFY_ROWS];
 bfloat beta[JUSTFIT_TARGET_VERIFY_ROWS];
};
#include "metal/kernels/common/compact_gdn.h"

static __attribute__((noinline)) float lq_tree_read(device const uchar *head,uint row,uint column,uint headIndex) {
 return compact_read(head,row,column,headIndex);
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth>
static inline void gdn_decode_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *conv_state_out,
    device bfloat *mixed_qkv, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    threadgroup GdnDecodeShared<HeadDim> &shared, uint value_head, uint lane,
    uint simd_group) {
  constexpr uint Tokens = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  static_assert(Tokens % kDecodeSimdgroups == 0 && HeadDim == 128,
                "one simdgroup per verify row, four channels per lane");
  const uint key_head = value_head / HeadsPerKey;
  // The key head's q/k rows and conv carry are shared by HeadsPerKey value
  // heads; the first of them writes the shared copies.
  const bool shared_writer = value_head % HeadsPerKey == 0;
  const bool carrier = simd_group < JUSTFIT_GDN_CONVOLUTION_TAPS - 1;
  for (uint token = simd_group; token < Tokens; token += kDecodeSimdgroups) {
  const uint q_channel = key_head * HeadDim + lane;
  const uint k_channel = KeyWidth + q_channel;
  const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;

  float q[Groups], k[Groups];
  bfloat v[Groups], carry_v[Groups], carry_q[Groups], carry_k[Groups];
  for (uint g = 0; g < Groups; ++g) {
    q[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               PackedWidth, ConvDim, token,
                               q_channel + 32 * g));
    k[g] = float(gdn_conv_silu(packed, conv_state_in, conv_weights,
                               PackedWidth, ConvDim, token,
                               k_channel + 32 * g));
    v[g] = gdn_conv_silu(packed, conv_state_in, conv_weights, PackedWidth,
                         ConvDim, token, v_channel + 32 * g);
    if (carrier) {
      carry_v[g] = gdn_conv_carry(packed, conv_state_in, PackedWidth, ConvDim,
                                  Tokens, simd_group, v_channel + 32 * g);
      carry_q[g] = shared_writer
          ? gdn_conv_carry(packed, conv_state_in, PackedWidth, ConvDim, Tokens,
                           simd_group, q_channel + 32 * g)
          : bfloat(0.0f);
      carry_k[g] = shared_writer
          ? gdn_conv_carry(packed, conv_state_in, PackedWidth, ConvDim, Tokens,
                           simd_group, k_channel + 32 * g)
          : bfloat(0.0f);
    }
  }
  GdnGates gates{};
  if (lane == 0)
    gates = gdn_gates(packed + token * PackedWidth, dt_bias, a_scale, BOffset,
                      AOffset, value_head);
  float q_sum = 0.0f, k_sum = 0.0f;
  for (uint g = 0; g < Groups; ++g) {
    q_sum += simd_sum(q[g] * q[g]);
    k_sum += simd_sum(k[g] * k[g]);
  }
  const float q_scale = rsqrt(q_sum / HeadDim + kRmsEpsilon);
  const float k_scale = rsqrt(k_sum / HeadDim + kRmsEpsilon);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
    const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
    shared.queries[token * HeadDim + dim] = query;
    shared.keys[token * HeadDim + dim] = key;
    shared.values[token * HeadDim + dim] = v[g];
    if (shared_writer)
      mixed_qkv[token * ConvDim + k_channel + 32 * g] = key;
    mixed_qkv[token * ConvDim + v_channel + 32 * g] = v[g];
  }
  if (lane == 0) {
    const uint gate_index = token * ValueHeads + value_head;
    beta[gate_index] = gates.beta;
    decay[gate_index] = gates.decay;
    shared.beta[token] = gates.beta;
    shared.decay[token] = gates.decay;
  }
  if (carrier) {
    const uint row = simd_group;
    for (uint g = 0; g < Groups; ++g) {
      conv_state_out[row * ConvDim + v_channel + 32 * g] = carry_v[g];
      if (shared_writer) {
        conv_state_out[row * ConvDim + q_channel + 32 * g] = carry_q[g];
        conv_state_out[row * ConvDim + k_channel + 32 * g] = carry_k[g];
      }
    }
  }
}

}
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, class W>
static inline void gdn_decode_gate(threadgroup GdnDecodeShared<HeadDim> &shared,
                            device const bfloat *packed,
                            device const W *norm_weight, device bfloat *hidden,
                            bool tiled, uint value_head, uint lane,
                            uint token) {
#pragma clang fp reassociate(off)
  constexpr uint Groups = HeadDim / 32;
  constexpr uint ZOffset = ConvDim;
  const ulong row = ulong(token) * ValueHeads;
  const ulong hidden_base =
      (row + gdn_output_head<KeyHeads, ValueHeads>(value_head, tiled)) *
      HeadDim;
  bfloat gate[Groups];
  W weight[Groups];
  float value[Groups];
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    gate[g] = packed[token * PackedWidth + ZOffset + value_head * HeadDim + dim];
    weight[g] = norm_weight[dim];
    value[g] = float(shared.rows[token * HeadDim + dim]);
  }
  float total = 0.0f;
  for (uint g = 0; g < Groups; ++g)
    total += simd_sum(value[g] * value[g]);
  const float inverse = rsqrt(total / HeadDim + kRmsEpsilon);
  for (uint g = 0; g < Groups; ++g) {
    const uint dim = 32 * g + lane;
    const bfloat normalized = bfloat((value[g] * inverse) * float(weight[g]));
    const float z = float(gate[g]);
    const bfloat gated = bfloat((float(normalized) * z) /
                                (1.0f + fast::exp2(-1.44269504089f * z)));
    hidden[hidden_base + dim] = gated;
    shared.rows[token * HeadDim + dim] = gated;
  }
}

static inline bfloat gdn_tree_conv_silu(device const bfloat *packed,
                                 device const bfloat *conv_state_in,
                                 device const bfloat *conv_weights,
                                 uint packed_width, uint conv_dim,
                                 constant JustFitTreeLane &tree, uint row,
                                 uint channel) {
  constexpr uint Taps = JUSTFIT_GDN_CONVOLUTION_TAPS, Carried = Taps - 1;
  const uint depth = tree.depth[row];
  constant uint *path = tree.path[tree.copy[row]];
  float value = 0.0f;
  for (uint tap = 0; tap < Taps; ++tap) {
    uint position = depth + tap;
    bfloat input = position < Carried
                       ? conv_state_in[position * conv_dim + channel]
                       : packed[path[position - Carried] * packed_width + channel];
    value += float(input) * float(conv_weights[channel * Taps + tap]);
  }
  value = float(bfloat(value));
  return bfloat(justfit_silu(value));
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth>
static inline void gdn_tree_prologue(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const bfloat *conv_state_in, device bfloat *mixed_qkv,
    device const float *a_scale, device const bfloat *dt_bias,
    device float *decay, device bfloat *beta, constant JustFitTreeLane &tree,
    threadgroup GdnDecodeShared<HeadDim> &shared, uint value_head, uint lane,
    uint simd_group) {
  constexpr uint Tokens = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint HeadsPerKey = ValueHeads / KeyHeads;
  constexpr uint KeyWidth = KeyHeads * HeadDim;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constexpr uint BOffset = ConvDim + ValueWidth;
  constexpr uint AOffset = BOffset + ValueHeads;
  constexpr uint Groups = HeadDim / 32;
  const uint key_head = value_head / HeadsPerKey;
  const bool shared_writer = value_head % HeadsPerKey == 0;
  for (uint token = simd_group; token < Tokens; token += kDecodeSimdgroups) {
    const uint q_channel = key_head * HeadDim + lane;
    const uint k_channel = KeyWidth + q_channel;
    const uint v_channel = 2 * KeyWidth + value_head * HeadDim + lane;
    float q[Groups], k[Groups];
    bfloat v[Groups];
    for (uint g = 0; g < Groups; ++g) {
      q[g] = float(gdn_tree_conv_silu(packed, conv_state_in, conv_weights, PackedWidth, ConvDim,
                                      tree, token, q_channel + 32 * g));
      k[g] = float(gdn_tree_conv_silu(packed, conv_state_in, conv_weights, PackedWidth, ConvDim,
                                      tree, token, k_channel + 32 * g));
      v[g] = gdn_tree_conv_silu(packed, conv_state_in, conv_weights, PackedWidth, ConvDim, tree,
                                token, v_channel + 32 * g);
    }
    GdnGates gates{};
    if (lane == 0)
      gates = gdn_gates(packed + token * PackedWidth, dt_bias, a_scale, BOffset,
                        AOffset, value_head);
    float q_sum = 0.0f, k_sum = 0.0f;
    for (uint g = 0; g < Groups; ++g) {
      q_sum += simd_sum(q[g] * q[g]);
      k_sum += simd_sum(k[g] * k[g]);
    }
    const float q_scale = rsqrt(q_sum / HeadDim + kRmsEpsilon);
    const float k_scale = rsqrt(k_sum / HeadDim + kRmsEpsilon);
    for (uint g = 0; g < Groups; ++g) {
      const uint dim = 32 * g + lane;
      const bfloat query = bfloat(float(bfloat(q[g] * q_scale)) * 0.0078125f);
      const bfloat key = bfloat(float(bfloat(k[g] * k_scale)) * 0.08838834765f);
      shared.queries[token * HeadDim + dim] = query;
      shared.keys[token * HeadDim + dim] = key;
      shared.values[token * HeadDim + dim] = v[g];
      if (shared_writer)
        mixed_qkv[token * ConvDim + k_channel + 32 * g] = key;
      mixed_qkv[token * ConvDim + v_channel + 32 * g] = v[g];
    }
    if (lane == 0) {
      const uint gate_index = token * ValueHeads + value_head;
      beta[gate_index] = gates.beta;
      decay[gate_index] = gates.decay;
      shared.beta[token] = gates.beta;
      shared.decay[token] = gates.decay;
    }
  }
}

template <uint HeadDim, uint RowsInFlight>
static inline void gdn_tree_step(thread float (&state)[RowsInFlight][4],
                          threadgroup GdnDecodeShared<HeadDim> &shared,
                          thread const uint (&value_dim)[RowsInFlight],
                          uint token, uint lane, bool output, device float *saved = nullptr) {
  const float d = shared.decay[token];
  const float b = float(shared.beta[token]);
  threadgroup const bfloat *key = shared.keys + token * HeadDim + lane * 4;
  threadgroup const bfloat *query = shared.queries + token * HeadDim + lane * 4;
  float memory[RowsInFlight];
  for (uint r = 0; r < RowsInFlight; ++r) {
    memory[r] = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] *= d;
      memory[r] += state[r][i] * float(key[i]);
    }
    memory[r] = simd_sum(memory[r]);
  }
  float result[RowsInFlight];
  for (uint r = 0; r < RowsInFlight; ++r) {
    const float delta =
        (float(shared.values[token * HeadDim + value_dim[r]]) - memory[r]) * b;
    if (saved && output && lane==0) saved[token*HeadDim+value_dim[r]]=delta;
    result[r] = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] += float(key[i]) * delta;
      result[r] += state[r][i] * float(query[i]);
    }
    result[r] = simd_sum(result[r]);
  }
  if (output && lane == 0) {
    for (uint r = 0; r < RowsInFlight; ++r)
      shared.rows[token * HeadDim + value_dim[r]] = bfloat(result[r]);
  }
}

template <uint HeadDim, uint RowsInFlight>
static inline void virtual_step(thread float (&state)[RowsInFlight][4],
                          threadgroup GdnDecodeShared<HeadDim> &shared,
                          thread const uint (&value_dim)[RowsInFlight],
                          thread float (&coef)[RowsInFlight], uint token, uint lane, bool output, device float *saved = nullptr) {
  const float d = shared.decay[token];
  const float b = float(shared.beta[token]);
  threadgroup const bfloat *key = shared.keys + token * HeadDim + lane * 4;
  threadgroup const bfloat *query = shared.queries + token * HeadDim + lane * 4;
  float memory[RowsInFlight];
  for (uint r = 0; r < RowsInFlight; ++r) {
    memory[r] = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] *= d;
      memory[r] += state[r][i] * float(key[i]);
    }
    coef[r] *= d;
      memory[r] = simd_sum(memory[r]+coef[r]*shared.projectK[token][lane]);
  }
  float result[RowsInFlight];
  for (uint r = 0; r < RowsInFlight; ++r) {
    const float delta =
        (float(shared.values[token * HeadDim + value_dim[r]]) - memory[r]) * b;
    if (saved && output && lane==0) saved[token*HeadDim+value_dim[r]]=delta;
    result[r] = 0.0f;
    for (uint i = 0; i < 4; ++i) {
      state[r][i] += float(key[i]) * delta;
      result[r] += state[r][i] * float(query[i]);
    }
    result[r] = simd_sum(result[r]+coef[r]*shared.projectQ[token][lane]);
  }
  if (output && lane == 0) {
    for (uint r = 0; r < RowsInFlight; ++r)
      shared.rows[token * HeadDim + value_dim[r]] = bfloat(result[r]);
  }
}

template <uint HeadDim, uint RowsInFlight>
static inline void gdn_tree_scan(device const float *state_in, constant JustFitTreeLane &tree,
                          threadgroup GdnDecodeShared<HeadDim> &shared,
                          uint value_head, uint lane, uint simd_group, device float *saved = nullptr) {
  constexpr uint Tokens = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint Batches = HeadDim / kDecodeSimdgroups;
  static_assert(Batches % RowsInFlight == 0, "rows in flight tile the head");
  device const uchar *cell=compact_head(state_in,value_head);
  const uint count=compact_count(cell);
  device const half4 *left=reinterpret_cast<device const half4*>(cell+17408);
  device const half4 *right=left+128;
  device const bfloat *keys=r94_keys(cell,value_head);
  device const float *updates=reinterpret_cast<device const float*>(cell+JUSTFIT_COMPACT_UPDATES);
  device const float *decays=reinterpret_cast<device const float*>(cell+JUSTFIT_COMPACT_DECAYS);
  device const float *scales=reinterpret_cast<device const float*>(cell+16384);
  float carry=1.f;
  for(uint t=0;t<count;++t)carry*=decays[t];
  float suffix=1.f;
  for(uint t=1;t<count;++t)if(lane>=4 && t>lane-4)suffix*=decays[t];
  for(uint component=0;component<4+count;++component){
    float pk=0,pq=0;
    for(uint i=0;i<4;++i){
      uint col=lane*4+i;
      float basis=component<4?float(right[col][component]):float(keys[(component-4)*128+col]);
      pk+=basis*float(shared.keys[simd_group*128+col]);
      pq+=basis*float(shared.queries[simd_group*128+col]);
    }
    pk=simd_sum(pk);pq=simd_sum(pq);
    if(lane==0){shared.projectK[simd_group][component]=pk;shared.projectQ[simd_group][component]=pq;}
  }
  if(lane>=4+count){shared.projectK[simd_group][lane]=0;shared.projectQ[simd_group][lane]=0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint batch = 0; batch < Batches; batch += RowsInFlight) {
    uint value_dim[RowsInFlight];
    float initial[RowsInFlight][4], state[RowsInFlight][4];
    float baseCoef[RowsInFlight],coef[RowsInFlight];
    for(uint r=0;r<RowsInFlight;++r){
      uint row=(batch+r)*kDecodeSimdgroups+simd_group;
      value_dim[r]=row;
      float4 codes=float4(reinterpret_cast<device const char4*>(cell)[row*32+lane]);
      float4 cols=reinterpret_cast<device const float4*>(scales)[lane];
      float4 boundary=codes*(1.f/127.f)*cols*scales[128+row]*carry;
      for(uint i=0;i<4;++i)initial[r][i]=state[r][i]=boundary[i];
      float factor=0;
      if(lane<4)factor=float(left[row][lane])*carry;
      else if(lane<4+count)factor=updates[(lane-4)*128+row]*suffix;
      baseCoef[r]=coef[r]=factor;
    }
    uint current = 0;
    for (uint token = 0; token < Tokens; ++token) {
      if (token && tree.parent[token] != current) {
        const uint parent = tree.parent[token];
        for(uint r=0;r<RowsInFlight;++r)coef[r]=baseCoef[r];
        constant uint *path = tree.path[tree.copy[parent]];
        for (uint r = 0; r < RowsInFlight; ++r)
          for (uint i = 0; i < 4; ++i)
            state[r][i] = initial[r][i];
        for (uint depth = 0; depth <= tree.depth[parent]; ++depth)
          virtual_step<HeadDim, RowsInFlight>(state, shared, value_dim, coef, path[depth], lane, false);
      }
      virtual_step<HeadDim, RowsInFlight>(state, shared, value_dim, coef, token, lane, true,saved);
      current = token;
    }
  }
}

template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W>
static inline void gdn_tree_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    device const W *gdn_norm_weight, device bfloat *gdn_hidden,
    constant GDNTreeDecodeParams &tree_params,
    uint2 group, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim> &shared,
    device bfloat *table, device float *sums, device uchar *next) {
  constexpr uint Rows = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constant GDNDecodeBatchParams &params = tree_params.base;
  uint batch = group.y;
  if (batch >= 2)
    return;
  constant JustFitTreeLane &tree = tree_params.tree.lane[batch];
  device const uchar *current = JUSTFIT_LANE_BINDING(batch, current0, current1, current2, current3);
  packed += ulong(batch) * Rows * PackedWidth;
  mixed += ulong(batch) * Rows * ConvDim;
  decay += ulong(batch) * Rows * ValueHeads;
  beta += ulong(batch) * Rows * ValueHeads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_tree_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      packed, conv_weights, conv_state_in, mixed, a_scale, dt_bias, decay, beta,
      tree, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  gdn_tree_scan<HeadDim, RowsInFlight>(state_in, tree, shared, group.x, lane, simd_group,
      reinterpret_cast<device float*>(next+params.convolution_state_bytes)+(params.layer*ValueHeads+group.x)*Rows*HeadDim);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tiled = params.tiled_heads != 0;
  for (uint row = simd_group; row < Rows; row += kDecodeSimdgroups) {
  gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      shared, packed, gdn_norm_weight, lane_hidden, tiled, group.x, lane,
      row);
  if (table) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = head * HeadDim + g * 64 + 2 * lane;
      const uint local = row * HeadDim + g * 64 + 2 * lane;
      Table::write(table + (ulong(batch) * Rows + row / 8 * 8) * ValueWidth,
                   sums + (ulong(batch) * (Rows/8) + row/8) * Table::sums_per_tile(ValueWidth), ValueWidth,
                   column / 64, row % 8, lane, shared.rows[local], shared.rows[local + 1]);
    }
  }
  }
}

template <uint HeadDim, uint RowsInFlight>
static inline void lq_chain_scan(device const float *state_in,
 threadgroup GdnDecodeShared<HeadDim> &shared,uint head,uint lane,uint sg) {
 for(uint batch=0;batch<16;batch+=RowsInFlight){
  uint rows[RowsInFlight];float state[RowsInFlight][4];
  for(uint r=0;r<RowsInFlight;r++){
   rows[r]=(batch+r)*8+sg;
   for(uint i=0;i<4;i++){
    volatile float carried=
#ifdef TREE_DENSE_REFERENCE
     state_in[(ulong(head)*128+rows[r])*128+lane*4+i];
#else
     lq_tree_read(compact_head(state_in,head),rows[r],lane*4+i,head);
#endif
    state[r][i]=carried;
   }
  }
  for(uint token=0;token<JUSTFIT_TARGET_VERIFY_ROWS;token++)
   gdn_tree_step<HeadDim,RowsInFlight>(state,shared,rows,token,lane,true);
 }
}
template <uint KeyHeads, uint ValueHeads, uint HeadDim, uint ConvDim,
          uint PackedWidth, uint RowsInFlight, class Table, class W>
static inline void lq_chain_batch_phase(
    device const bfloat *packed, device const bfloat *conv_weights,
    device const uchar *current0, device const uchar *current1,
    device const uchar *current2, device const uchar *current3,
    device bfloat *unused_conv, device bfloat *mixed, device const float *a_scale,
    device const bfloat *dt_bias, device float *decay, device bfloat *beta,
    device const W *gdn_norm_weight, device bfloat *gdn_hidden,
    constant GDNDecodeBatchParams &tree_params,
    uint2 group, uint lane, uint simd_group,
    threadgroup GdnDecodeShared<HeadDim> &shared,
    device bfloat *table, device float *sums) {
  constexpr uint Rows = JUSTFIT_TARGET_VERIFY_ROWS;
  constexpr uint ValueWidth = ValueHeads * HeadDim;
  constant GDNDecodeBatchParams &params = tree_params;
  uint batch = group.y;
  if (batch >= 2)
    return;
  device const uchar *current = JUSTFIT_LANE_BINDING(batch, current0, current1, current2, current3);
  packed += ulong(batch) * Rows * PackedWidth;
  mixed += ulong(batch) * Rows * ConvDim;
  decay += ulong(batch) * Rows * ValueHeads;
  beta += ulong(batch) * Rows * ValueHeads;
  device const bfloat *conv_state_in =
      reinterpret_cast<device const bfloat *>(
          current + ulong(params.layer) * params.conv_layer_bytes);
  device const float *state_in = reinterpret_cast<device const float *>(
      current + params.convolution_state_bytes +
      ulong(params.layer) * params.recurrent_layer_bytes);
  device bfloat *conv_state_out=unused_conv;
  device bfloat *lane_hidden = gdn_hidden + ulong(batch) * Rows * ValueWidth;
  gdn_decode_prologue<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      packed, conv_weights, conv_state_in, conv_state_out, mixed, a_scale, dt_bias, decay, beta, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  lq_chain_scan<HeadDim, RowsInFlight>(state_in, shared, group.x, lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const bool tiled = params.tiled_heads != 0;
  for (uint row = simd_group; row < Rows; row += kDecodeSimdgroups) {
  gdn_decode_gate<KeyHeads, ValueHeads, HeadDim, ConvDim, PackedWidth>(
      shared, packed, gdn_norm_weight, lane_hidden, tiled, group.x, lane,
      row);
  if (table) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint head = gdn_output_head<KeyHeads, ValueHeads>(group.x, tiled);
    for (uint g = 0; g < HeadDim / 64; ++g) {
      const uint column = head * HeadDim + g * 64 + 2 * lane;
      const uint local = row * HeadDim + g * 64 + 2 * lane;
      Table::write(table + (ulong(batch) * Rows + row / 8 * 8) * ValueWidth,
                   sums + (ulong(batch) * (Rows/8) + row/8) * Table::sums_per_tile(ValueWidth), ValueWidth,
                   column / 64, row % 8, lane, shared.rows[local], shared.rows[local + 1]);
    }
  }
  }
}
kernel void compact_gdn_tree_verify(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const bfloat *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],
 constant GDNTreeDecodeParams &params [[buffer(17)]],
 uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 device uchar *next = JUSTFIT_LANE_BINDING(group.y,next0,next1,next2,next3);
 threadgroup GdnDecodeShared<128> shared;
 gdn_tree_batch_phase<16,48,128,10240,16640,2,q4sg::Table64,bfloat>(
 packed,conv_weights,current0,current1,current2,current3,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,
 nullptr,nullptr,next);
}

kernel void compact_gdn_tree_verify_table64(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const bfloat *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],device bfloat *table [[buffer(17)]], device float *sums [[buffer(18)]],
 constant GDNTreeDecodeParams &params [[buffer(19)]],
 uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 device uchar *next = JUSTFIT_LANE_BINDING(group.y,next0,next1,next2,next3);
 threadgroup GdnDecodeShared<128> shared;
 gdn_tree_batch_phase<16,48,128,10240,16640,2,q4sg::Table64,bfloat>(
 packed,conv_weights,current0,current1,current2,current3,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,
 table,sums,next);
}

kernel void compact_gdn_tree_chain_verify(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const bfloat *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],
 constant GDNDecodeBatchParams &params [[buffer(17)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]){
 (void)next2;(void)next3;
 device uchar *next=group.y==0?next0:next1;
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(params.layer)*params.conv_layer_bytes);
 threadgroup GdnDecodeShared<128> shared;
 lq_chain_batch_phase<16,48,128,10240,16640,2,q4sg::Table64,bfloat>(
 packed,conv_weights,current0,current1,current2,current3,convOut,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,nullptr,nullptr);
}

kernel void compact_gdn_tree_chain_verify_table64(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const bfloat *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],device bfloat *table [[buffer(17)]], device float *sums [[buffer(18)]],
 constant GDNDecodeBatchParams &params [[buffer(19)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]){
 (void)next2;(void)next3;
 device uchar *next=group.y==0?next0:next1;
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(params.layer)*params.conv_layer_bytes);
 threadgroup GdnDecodeShared<128> shared;
 lq_chain_batch_phase<16,48,128,10240,16640,2,q4sg::Table64,bfloat>(
 packed,conv_weights,current0,current1,current2,current3,convOut,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,table,sums);
}

// The accepted path has already been gathered to consecutive rows by Runtime.
// All reads of the old compact cell finish before any tail metadata is changed.
kernel void compact_gdn_tree_commit(
 device const bfloat *packed [[buffer(0)]], device const bfloat *mixed [[buffer(1)]],
 device const float *decay [[buffer(2)]], device const bfloat *beta [[buffer(3)]],
 device const uchar *current0 [[buffer(4)]], device const uchar *current1 [[buffer(5)]],
 device const uchar *current2 [[buffer(6)]], device const uchar *current3 [[buffer(7)]],
 device uchar *next0 [[buffer(8)]], device uchar *next1 [[buffer(9)]],
 device uchar *next2 [[buffer(10)]], device uchar *next3 [[buffer(11)]],
 device const uint *retained [[buffer(12)]], device const uint *accepted_path [[buffer(13)]],
 constant GDNBatchCommitParams &params [[buffer(14)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 constexpr uint Rows=JUSTFIT_TARGET_VERIFY_ROWS,ConvDim=10240,PackedWidth=16640;
 const uint head=group.x,layer=group.y,batch=group.z,count=retained[batch];
 device const uchar *current=JUSTFIT_LANE_BINDING(batch,current0,current1,current2,current3);
 device uchar *next=JUSTFIT_LANE_BINDING(batch,next0,next1,next2,next3);
 device uchar *h=const_cast<device uchar*>(current+params.convolution_state_bytes+
   ulong(layer)*params.recurrent_layer_bytes+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES);
 const uint start=compact_count(h);
 // Host/acceptance guarantees <=8 retained and starts <=8. Never truncate silently.
 if(count>8 || start+count>16) return;
 const ulong base=ulong(layer)*32+ulong(batch)*Rows;
 packed+=base*PackedWidth; mixed+=base*ConvDim; decay+=base*48; (void)beta;(void)lane;(void)sg;
 device const bfloat *convIn=reinterpret_cast<device const bfloat*>(current+ulong(layer)*params.conv_layer_bytes);
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(layer)*params.conv_layer_bytes);
 device const float *staged=reinterpret_cast<device const float*>(next+params.convolution_state_bytes)
                         +(layer*48+head)*8*128;
 for(uint element=head*256+tid;element<3*ConvDim;element+=48*256){
  uint row=element/ConvDim,channel=element%ConvDim;
  convOut[element]=gdn_conv_carry(packed,convIn,PackedWidth,ConvDim,count,row,channel);
 }
 for(uint token=0;token<count;token++){
  if(tid<128){
   if(head%3==0) r94_keys(h,head)[(start+token)*128+tid]=
    bfloat(mixed[ulong(token)*ConvDim+2048+(head/3)*128+tid]);
   reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_UPDATES)[(start+token)*128+tid]=staged[accepted_path[batch*JUSTFIT_TREE_MAXIMUM_DEPTH+token]*128+tid];
  }
  if(tid==0)reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_DECAYS)[start+token]=decay[token*48+head];
 }
 threadgroup_barrier(mem_flags::mem_device);
 if(tid==0)*reinterpret_cast<device uint*>(h+JUSTFIT_COMPACT_COUNT)=start+count;
}
// The accepted path has already been gathered to consecutive rows by Runtime.
// All reads of the old compact cell finish before any tail metadata is changed.
kernel void compact_gdn_tree_chain_commit0(
 device const bfloat *packed [[buffer(0)]], device const bfloat *mixed [[buffer(1)]],
 device const float *decay [[buffer(2)]], device const bfloat *beta [[buffer(3)]],
 device const uchar *current0 [[buffer(4)]], device const uchar *current1 [[buffer(5)]],
 device const uchar *current2 [[buffer(6)]], device const uchar *current3 [[buffer(7)]],
 device uchar *next0 [[buffer(8)]], device uchar *next1 [[buffer(9)]],
 device uchar *next2 [[buffer(10)]], device uchar *next3 [[buffer(11)]],
 device const uint *retained [[buffer(12)]], constant GDNBatchCommitParams &params [[buffer(13)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 constexpr uint Rows=JUSTFIT_TARGET_VERIFY_ROWS,ConvDim=10240,PackedWidth=16640;
 const uint head=group.x,layer=group.y,batch=group.z,total=retained[batch];
 device const uchar *current=JUSTFIT_LANE_BINDING(batch,current0,current1,current2,current3);
 device uchar *next=JUSTFIT_LANE_BINDING(batch,next0,next1,next2,next3);
 device uchar *h=const_cast<device uchar*>(current+params.convolution_state_bytes+
   ulong(layer)*params.recurrent_layer_bytes+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES);
 const uint start=compact_count(h);
 // Host/acceptance guarantees <=8 retained and starts <=8. Never truncate silently.
 if(total>16 || start>16) return;
 device uint *offset=reinterpret_cast<device uint*>(next+params.convolution_state_bytes+
   ulong(layer)*48*8*128*4+ulong(head)*8*128*4);
 const uint skip=0;
 const uint count=min(total,16u-start);
 if(tid==0)*offset=count;
 const ulong base=ulong(layer)*32+ulong(batch)*Rows;
 packed+=base*PackedWidth; mixed+=base*ConvDim; decay+=base*48; beta+=base*48;
 device const bfloat *convIn=reinterpret_cast<device const bfloat*>(current+ulong(layer)*params.conv_layer_bytes);
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(layer)*params.conv_layer_bytes);
 threadgroup float updates[16*128];
 for(uint row=sg;row<128;row+=8) {
  float state[4];for(uint i=0;i<4;i++)state[i]=lq_tree_read(h,row,lane*4+i,head);
  for(uint token=0;token<count;token++) {
   float memory=0;float key[4];
   for(uint i=0;i<4;i++){
    key[i]=float(mixed[ulong(token+skip)*ConvDim+2048+(head/3)*128+lane*4+i]);
    state[i]*=decay[(token+skip)*48+head];memory+=state[i]*key[i];
   }
   memory=simd_sum(memory);
   const float delta=(float(mixed[ulong(token+skip)*ConvDim+4096+head*128+row])-memory)*float(beta[(token+skip)*48+head]);
   for(uint i=0;i<4;i++)state[i]+=key[i]*delta;
   if(lane==0)updates[token*128+row]=delta;
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 for(uint element=head*256+tid;element<3*ConvDim;element+=48*256){
  uint row=element/ConvDim,channel=element%ConvDim;
  convOut[element]=gdn_conv_carry(packed,convIn,PackedWidth,ConvDim,total,row,channel);
 }
 for(uint token=0;token<count;token++){
  if(tid<128){
   if(head%3==0) r94_keys(h,head)[(start+token)*128+tid]=
    bfloat(mixed[ulong(token+skip)*ConvDim+2048+(head/3)*128+tid]);
   reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_UPDATES)[(start+token)*128+tid]=updates[token*128+tid];
  }
  if(tid==0)reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_DECAYS)[start+token]=decay[(token+skip)*48+head];
 }
 threadgroup_barrier(mem_flags::mem_device);
 if(tid==0)*reinterpret_cast<device uint*>(h+JUSTFIT_COMPACT_COUNT)=start+count;
}

// The accepted path has already been gathered to consecutive rows by Runtime.
// All reads of the old compact cell finish before any tail metadata is changed.
kernel void compact_gdn_tree_chain_commit1(
 device const bfloat *packed [[buffer(0)]], device const bfloat *mixed [[buffer(1)]],
 device const float *decay [[buffer(2)]], device const bfloat *beta [[buffer(3)]],
 device const uchar *current0 [[buffer(4)]], device const uchar *current1 [[buffer(5)]],
 device const uchar *current2 [[buffer(6)]], device const uchar *current3 [[buffer(7)]],
 device uchar *next0 [[buffer(8)]], device uchar *next1 [[buffer(9)]],
 device uchar *next2 [[buffer(10)]], device uchar *next3 [[buffer(11)]],
 device const uint *retained [[buffer(12)]], constant GDNBatchCommitParams &params [[buffer(13)]],
 uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 constexpr uint Rows=JUSTFIT_TARGET_VERIFY_ROWS,ConvDim=10240,PackedWidth=16640;
 const uint head=group.x,layer=group.y,batch=group.z,total=retained[batch];
 device const uchar *current=JUSTFIT_LANE_BINDING(batch,current0,current1,current2,current3);
 device uchar *next=JUSTFIT_LANE_BINDING(batch,next0,next1,next2,next3);
 device uchar *h=const_cast<device uchar*>(current+params.convolution_state_bytes+
   ulong(layer)*params.recurrent_layer_bytes+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES);
 const uint start=compact_count(h);
 // Host/acceptance guarantees <=8 retained and starts <=8. Never truncate silently.
 if(total>16 || start>16) return;
 device uint *offset=reinterpret_cast<device uint*>(next+params.convolution_state_bytes+
   ulong(layer)*48*8*128*4+ulong(head)*8*128*4);
 const uint skip=*offset;
 const uint count=total-skip;
 if(count==0)return;
 const ulong base=ulong(layer)*32+ulong(batch)*Rows;
 packed+=base*PackedWidth; mixed+=base*ConvDim; decay+=base*48; beta+=base*48;
 device const bfloat *convIn=reinterpret_cast<device const bfloat*>(current+ulong(layer)*params.conv_layer_bytes);
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(layer)*params.conv_layer_bytes);
 threadgroup float updates[16*128];
 for(uint row=sg;row<128;row+=8) {
  float state[4];for(uint i=0;i<4;i++)state[i]=lq_tree_read(h,row,lane*4+i,head);
  for(uint token=0;token<count;token++) {
   float memory=0;float key[4];
   for(uint i=0;i<4;i++){
    key[i]=float(mixed[ulong(token+skip)*ConvDim+2048+(head/3)*128+lane*4+i]);
    state[i]*=decay[(token+skip)*48+head];memory+=state[i]*key[i];
   }
   memory=simd_sum(memory);
   const float delta=(float(mixed[ulong(token+skip)*ConvDim+4096+head*128+row])-memory)*float(beta[(token+skip)*48+head]);
   for(uint i=0;i<4;i++)state[i]+=key[i]*delta;
   if(lane==0)updates[token*128+row]=delta;
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 for(uint element=3*ConvDim;element<3*ConvDim;element+=48*256){
  uint row=element/ConvDim,channel=element%ConvDim;
  convOut[element]=gdn_conv_carry(packed,convIn,PackedWidth,ConvDim,count,row,channel);
 }
 for(uint token=0;token<count;token++){
  if(tid<128){
   if(head%3==0) r94_keys(h,head)[(start+token)*128+tid]=
    bfloat(mixed[ulong(token+skip)*ConvDim+2048+(head/3)*128+tid]);
   reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_UPDATES)[(start+token)*128+tid]=updates[token*128+tid];
  }
  if(tid==0)reinterpret_cast<device float*>(h+JUSTFIT_COMPACT_DECAYS)[start+token]=decay[(token+skip)*48+head];
 }
 threadgroup_barrier(mem_flags::mem_device);
 if(tid==0)*reinterpret_cast<device uint*>(h+JUSTFIT_COMPACT_COUNT)=start+count;
}

// Test seam: identical production tree recurrence with prepared real-shaped rows.

kernel void compact_gdn_tree_verify_f32(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const float *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],
 constant GDNTreeDecodeParams &params [[buffer(17)]],
 uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 device uchar *next = JUSTFIT_LANE_BINDING(group.y,next0,next1,next2,next3);
 threadgroup GdnDecodeShared<128> shared;
 gdn_tree_batch_phase<16,48,128,10240,16640,2,gguf_sg::Table16,float>(
 packed,conv_weights,current0,current1,current2,current3,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,
 nullptr,nullptr,next);
}

kernel void compact_gdn_tree_verify_table16_f32(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const float *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],device bfloat *table [[buffer(17)]], device float *sums [[buffer(18)]],
 constant GDNTreeDecodeParams &params [[buffer(19)]],
 uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
 device uchar *next = JUSTFIT_LANE_BINDING(group.y,next0,next1,next2,next3);
 threadgroup GdnDecodeShared<128> shared;
 gdn_tree_batch_phase<16,48,128,10240,16640,2,gguf_sg::Table16,float>(
 packed,conv_weights,current0,current1,current2,current3,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,
 table,sums,next);
}

kernel void compact_gdn_tree_chain_verify_f32(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const float *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],
 constant GDNDecodeBatchParams &params [[buffer(17)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]){
 (void)next2;(void)next3;
 device uchar *next=group.y==0?next0:next1;
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(params.layer)*params.conv_layer_bytes);
 threadgroup GdnDecodeShared<128> shared;
 lq_chain_batch_phase<16,48,128,10240,16640,2,gguf_sg::Table16,float>(
 packed,conv_weights,current0,current1,current2,current3,convOut,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,nullptr,nullptr);
}

kernel void compact_gdn_tree_chain_verify_table16_f32(device const bfloat *packed [[buffer(0)]],
 device const bfloat *conv_weights [[buffer(1)]],
 device const uchar *current0 [[buffer(2)]], device const uchar *current1 [[buffer(3)]],
 device const uchar *current2 [[buffer(4)]], device const uchar *current3 [[buffer(5)]],
 device uchar *next0 [[buffer(6)]], device uchar *next1 [[buffer(7)]],
 device uchar *next2 [[buffer(8)]], device uchar *next3 [[buffer(9)]],
 device bfloat *mixed [[buffer(10)]], device const float *a_scale [[buffer(11)]],
 device const bfloat *dt_bias [[buffer(12)]], device float *decay [[buffer(13)]],
 device bfloat *beta [[buffer(14)]], device const float *norm [[buffer(15)]],
 device bfloat *hidden [[buffer(16)]],device bfloat *table [[buffer(17)]], device float *sums [[buffer(18)]],
 constant GDNDecodeBatchParams &params [[buffer(19)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]){
 (void)next2;(void)next3;
 device uchar *next=group.y==0?next0:next1;
 device bfloat *convOut=reinterpret_cast<device bfloat*>(next+ulong(params.layer)*params.conv_layer_bytes);
 threadgroup GdnDecodeShared<128> shared;
 lq_chain_batch_phase<16,48,128,10240,16640,2,gguf_sg::Table16,float>(
 packed,conv_weights,current0,current1,current2,current3,convOut,mixed,a_scale,dt_bias,
 decay,beta,norm,hidden,params,group,lane,sg,shared,table,sums);
}
