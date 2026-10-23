// Modified for JustFit: product identity and integration.
#pragma once
#include <metal_stdlib>
using namespace metal;

// SiLU and the logistic sigmoid of every kernel, e^-x taken as
// fast::exp2(-x * log2(e)). A gated product multiplies justfit_silu(gate)
// into its other factor, so silu(g) * up evaluates g / (1 + e^-g) * up.
inline float justfit_silu(float x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float2 justfit_silu(float2 x) {
  return x / (1.0f + fast::exp2(-1.44269504089f * x));
}
inline float justfit_sigmoid(float x) {
  return 1.0f / (1.0f + fast::exp2(-1.44269504089f * x));
}
