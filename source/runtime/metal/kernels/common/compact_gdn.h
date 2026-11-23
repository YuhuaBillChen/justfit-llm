#pragma once
#include "metal/abi/CompactGDN.h"
inline device const uchar *compact_head(device const float *layer,uint head) {
 return reinterpret_cast<device const uchar*>(layer)+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES;
}
inline device uchar *compact_head(device float *layer,uint head) {
 return reinterpret_cast<device uchar*>(layer)+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES;
}
inline uint compact_count(device const uchar *head){return *reinterpret_cast<device const uint*>(head+JUSTFIT_COMPACT_COUNT);}
inline float compact_read(device const uchar *h,uint v,uint k,uint headIndex) {
 device const char *codes=reinterpret_cast<device const char*>(h);
 device const float *scales=reinterpret_cast<device const float*>(h+16384);
 device const half *factors=reinterpret_cast<device const half*>(h+17408);
 float x=float(codes[v*128+k])*(scales[k]*scales[128+v]/127);
 for(uint r=0;r<4;r++)x+=float(factors[v*4+r])*float(factors[512+k*4+r]);
 device const bfloat *keys=r94_keys(h,headIndex);
 device const float *updates=reinterpret_cast<device const float*>(h+JUSTFIT_COMPACT_UPDATES);
 device const float *decays=reinterpret_cast<device const float*>(h+JUSTFIT_COMPACT_DECAYS);
 const uint count=compact_count(h);
 // Evaluate the same per-coordinate decay/update chain as a dense state.
 for(uint t=0;t<count;t++)x=x*decays[t]+updates[t*128+v]*keys[t*128+k];
 return x;
}
