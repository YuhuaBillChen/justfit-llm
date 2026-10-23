// Modified for JustFit: product identity and integration.
#include "metal/kernels/common/activation.h"
kernel void cpu_up_silu_sum(device bfloat *output [[buffer(0)]], device const bfloat *gate [[buffer(1)]], device float *sums [[buffer(2)]], constant uint2 &shape [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
 const uint width=shape.x, first=shape.y, quant=(first/64)+group.y;
 for(uint row=sg;row<32;row+=8) {
  const ulong origin=ulong(group.x*32+row)*width+quant*64+lane;
  const bfloat y0=bfloat(justfit_silu(float(gate[origin]))*float(output[origin]));
  const bfloat y1=bfloat(justfit_silu(float(gate[origin+32]))*float(output[origin+32]));
  output[origin]=y0;output[origin+32]=y1;
  const float sum=simd_sum(float(y0)+float(y1));
  if(lane==0)sums[(ulong(group.x)*(width/64)+quant)*32+row]=sum;
 }
}
