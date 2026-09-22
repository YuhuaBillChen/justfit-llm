#include "metal/kernels/common/tq4_paging.h"

kernel void tq4_page_format_oracle(device const uint *packed [[buffer(0)]],
                                   device float *output [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index >= SplashTQ4HeadDimension)
    return;
  output[index] = SplashTQ4Codebook[splash_tq4_extract(packed, 0, index)] *
      splash_tq4_sign(false, index);
}
