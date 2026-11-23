#pragma once
#define JUSTFIT_GDN_COMPACT 1
#define JUSTFIT_COMPACT_HEAD_BYTES 27728u

#define JUSTFIT_COMPACT_UPDATES 19456u
#define JUSTFIT_COMPACT_DECAYS 27648u
#define JUSTFIT_COMPACT_COUNT 27712u
struct CompactLayerParams { uint32_t heads; uint32_t only_full; };

#ifdef __METAL_VERSION__
inline device const bfloat *r94_keys(device const uchar *cell,uint head){
 return reinterpret_cast<device const bfloat*>(cell+(48-head)*JUSTFIT_COMPACT_HEAD_BYTES)+(head/3)*2048;
}
inline device bfloat *r94_keys(device uchar *cell,uint head){
 return reinterpret_cast<device bfloat*>(cell+(48-head)*JUSTFIT_COMPACT_HEAD_BYTES)+(head/3)*2048;
}
#endif
