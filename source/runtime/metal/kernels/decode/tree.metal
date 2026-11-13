// Modified for JustFit: draft-tree verification (experiment).
#include "metal/abi/KernelABI.h"

// Greedy acceptance of a draft tree, one thread per lane. The target's token
// after each verify row is in target_tokens (the verify policy's argmax).
// From the anchor, the walk takes the child whose token is the target's token
// after the current row, as the chain takes its next draft token. The
// accepted path's target tokens are written back in chain order, so the
// lane's output reads as a chain cycle that accepted the path; the rows of
// the path go to path_rows for the gathers.
// Sampled lanes (JUSTFIT_TREE_SAMPLED): rows 0 .. chain_rows - 1 are the
// draft's sampled chain. On the chain, row r's child r + 1 is accepted by the
// chain's test (uniform * q < p, as accept_sampled_lane). Otherwise the walk
// takes row r's draw: on the chain a draw from the residual distribution, the
// correction the chain would emit; off the chain (and at the chain's end) a
// draw from the target distribution with the row's own uniform. When another
// child of the row holds the drawn token the walk continues there, else the
// draw is the cycle's last token. Every emitted token is a draw of the
// target's distribution after its prefix, as in the chain.
kernel void verify_tree_accept(device uint *target_tokens [[buffer(0)]],
                               device uint *retained [[buffer(1)]],
                               device uint *accepted_count [[buffer(2)]],
                               device uint *path_rows [[buffer(3)]],
                               device const TargetVocabularyRow *target_rows [[buffer(4)]],
                               device const float *uniforms [[buffer(5)]],
                               constant JustFitTreeAcceptParams &params [[buffer(6)]],
                               uint batch [[threadgroup_position_in_grid]],
                               uint thread_index [[thread_index_in_threadgroup]]) {
  if (thread_index != 0 || batch >= 2)
    return;
  constant JustFitTreeLane &tree = params.tree.lane[batch];
  device uint *tokens = target_tokens + batch * JUSTFIT_TARGET_VERIFY_ROWS;
  const bool sampled = (params.sampling_mask & (1u << batch)) != 0;
  const uint chain_rows = params.chain_rows[batch];
  device const TargetVocabularyRow *rows = target_rows + batch * JUSTFIT_TARGET_VERIFY_ROWS;
  device const float *lane_uniforms = uniforms + batch * JUSTFIT_SAMPLING_UNIFORMS;
  uint path[JUSTFIT_TREE_MAXIMUM_DEPTH];
  uint length = 1;
  path[0] = 0;
  uint current = 0;
  while (length < JUSTFIT_TREE_MAXIMUM_DEPTH) {
    if (sampled && current + 1 < chain_rows) {
      const float q = params.chain_q[batch][current];
      const float p = rows[current].draft_probability;
      if (lane_uniforms[JUSTFIT_UNIFORM_ACCEPTANCE + current] * q < p) {
        path[length++] = current + 1;
        current = current + 1;
        continue;
      }
    }
    const uint wanted = tokens[current];
    uint found = 0;
    for (uint row = current + 1; row < tree.rows; ++row) {
      if (tree.parent[row] == current && tree.token[row] == wanted &&
          !(sampled && row == current + 1 && current + 1 < chain_rows)) {
        found = row;
        break;
      }
    }
    if (!found)
      break;
    path[length++] = found;
    current = found;
  }
  uint output[JUSTFIT_TREE_MAXIMUM_DEPTH];
  // Each accepted row emits its child on the path; the last row its draw.
  for (uint i = 0; i < length; ++i)
    output[i] = i + 1 < length ? tree.token[path[i + 1]] : tokens[path[i]];
  for (uint i = 0; i < length; ++i)
    tokens[i] = output[i];
  // finish_acceptance (sampling.metal): the budget, then the first stop token.
  const uint accepted = length - 1;
  uint kept = min(accepted + 1, params.remaining[batch]);
  for (uint i = 0; i < kept; ++i) {
    if (tokens[i] == params.stop_token_0 || tokens[i] == params.stop_token_1) {
      kept = i + 1;
      break;
    }
  }
  accepted_count[batch] = accepted;
  retained[batch] = kept;
  for (uint i = 0; i < JUSTFIT_TREE_MAXIMUM_DEPTH; ++i)
    path_rows[batch * JUSTFIT_TREE_MAXIMUM_DEPTH + i] = i < length ? path[i] : path[length - 1];
}

// Moves the retained rows of the accepted path to rows 0 .. retained - 1 of
// one tensor family, in place. Each thread owns one column of one layer and
// lane and copies its rows in path order: path rows ascend and path[i] >= i,
// so no row is overwritten before it is read.
kernel void verify_tree_gather(device ushort *data [[buffer(0)]],
                               device const uint *path_rows [[buffer(1)]],
                               device const uint *retained [[buffer(2)]],
                               constant JustFitTreeGatherParams &params [[buffer(3)]],
                               uint3 position [[thread_position_in_grid]]) {
  const uint column = position.x, layer = position.y, lane = position.z;
  if (column >= params.columns || layer >= params.layers || lane >= params.lanes)
    return;
  const ulong base = ulong(layer) * params.layer_stride + ulong(lane) * params.lane_stride +
                     ulong(column / params.inner) * params.outer_stride +
                     ulong(column % params.inner) * params.inner_stride;
  const uint rows = retained[lane];
  for (uint i = 0; i < rows; ++i) {
    const uint source = path_rows[lane * JUSTFIT_TREE_MAXIMUM_DEPTH + i];
    if (source != i)
      data[base + ulong(i) * params.row_stride] = data[base + ulong(source) * params.row_stride];
  }
}
