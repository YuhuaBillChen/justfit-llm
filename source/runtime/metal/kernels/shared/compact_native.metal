// Lane zero initializes the shared usable flag before a threadgroup barrier.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#include <metal_stdlib>
using namespace metal;

// Independent Metal implementation of upstream's window-boundary representation.
// One 128-thread group per [V=128,K=128] head, rank 4, one power iteration.
static inline void orth(threadgroup float *x, threadgroup float *sums,
                 uint t, uint simd, uint lane) {
  for (uint r=0; r<4; ++r) {
    // Pilot: single-pass MGS mirrors upstream make_quantize; separately qualified recipe.
    for (uint pass=0; pass<1; ++pass) for (uint c=0; c<r; ++c) {
      float d=simd_sum(x[t*4+c]*x[t*4+r]);
      if (lane==0) sums[simd]=d;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      d=sums[0]+sums[1]+sums[2]+sums[3];
      x[t*4+r]-=d*x[t*4+c];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float d=simd_sum(x[t*4+r]*x[t*4+r]);
    if (lane==0) sums[simd]=d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    d=rsqrt(max(sums[0]+sums[1]+sums[2]+sums[3],1e-30f));
    x[t*4+r]*=d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#include "metal/abi/CompactGDN.h"
#include "metal/abi/CompactGDN.h"
static inline uint compact_count(device const uchar *head){return *reinterpret_cast<device const uint*>(head+JUSTFIT_COMPACT_COUNT);}
static inline float compact_read(device const uchar *h,uint v,uint k,uint headIndex) {
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

kernel void compact_unfold_layer(device const uchar *cell [[buffer(0)]],device float *out [[buffer(1)]],
 constant CompactLayerParams &params [[buffer(2)]],uint h [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]) {
 device const uchar *head=cell+ulong(h)*JUSTFIT_COMPACT_HEAD_BYTES;
 if(params.only_full && compact_count(head)!=16)return;
 for(uint i=t;i<16384;i+=128)out[ulong(h)*16384+i]=compact_read(head,i/128,i%128,h);
}
kernel void compact_pack_layer(device const float *all [[buffer(0)]],
                         device const float *q0 [[buffer(1)]],
                         device uchar *cell [[buffer(2)]],
                         constant CompactLayerParams &params [[buffer(3)]],
                         uint head [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  device uchar *h=cell+ulong(head)*JUSTFIT_COMPACT_HEAD_BYTES;
  if(params.only_full && *reinterpret_cast<device const uint*>(h+JUSTFIT_COMPACT_COUNT)!=16)return;
  device char *codes=reinterpret_cast<device char*>(h);
  device float *scales=reinterpret_cast<device float*>(h+16384);
  device half *factors=reinterpret_cast<device half*>(h+17408);
  threadgroup_barrier(mem_flags::mem_device);
  if(t==0)*reinterpret_cast<device uint*>(h+JUSTFIT_COMPACT_COUNT)=0;
  threadgroup float p[128*4], q[128*4], sk[128], sv[128], sums[4];
  device const float *s=all+ulong(head)*16384;
  for (uint r=0;r<4;++r) {
    float a=0;
    for(uint v=0;v<128;++v) a+=s[v*128+t]*q0[v*4+r];
    p[t*4+r]=a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  orth(p,sums,t,simd,lane);
  for(uint r=0;r<4;++r) {
    float a=0;
    for(uint k=0;k<128;++k) a+=s[t*128+k]*p[k*4+r];
    q[t*4+r]=a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  orth(q,sums,t,simd,lane);
  // q remains FP32 until every U column has consumed it.
  float u[4];
  for(uint r=0;r<4;++r) {
    float a=0;
    for(uint v=0;v<128;++v) a+=s[v*128+t]*q[v*4+r];
    u[r]=float(half(a));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint r=0;r<4;++r) {p[t*4+r]=u[r];q[t*4+r]=float(half(q[t*4+r]));}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean=0;
  for(uint v=0;v<128;++v) {
    float a=s[v*128+t];
    for(uint r=0;r<4;++r) a-=q[v*4+r]*p[t*4+r];
    mean+=abs(a);
  }
  sk[t]=max(sqrt(mean/128),1e-12f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float largest=0;
  for(uint k=0;k<128;++k) {
    float a=s[t*128+k];
    for(uint r=0;r<4;++r) a-=q[t*4+r]*p[k*4+r];
    largest=max(largest,abs(a/sk[k]));
  }
  sv[t]=max(largest,1e-12f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  scales[t]=sk[t];
  scales[128+t]=sv[t];
  for(uint r=0;r<4;++r) {
    factors[t*4+r]=half(q[t*4+r]);
    factors[512+t*4+r]=half(p[t*4+r]);
  }
  for(uint v=0;v<128;++v) {
    float a=s[v*128+t];
    for(uint r=0;r<4;++r) a-=q[v*4+r]*p[t*4+r];
    codes[v*128+t]=char(clamp(rint(a/(sk[t]*sv[v])*127),-127.0f,127.0f));
  }
  // The resident right factors p[k*4+r] already include FP16 rounding.
  // Normalize their columns before the shared 4x4 Cholesky solve.
  threadgroup float invnorm[4], lower[16];
  threadgroup uint usable;
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if(t<4) {
    float ss=0;
    for(uint k=0;k<128;++k) ss+=p[k*4+t]*p[k*4+t];
    invnorm[t]=ss>0 ? rsqrt(ss) : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(t<16) {
    uint i=t/4,j=t%4;
    float dot=0;
    for(uint k=0;k<128;++k) dot+=(p[k*4+i]*invnorm[i])*(p[k*4+j]*invnorm[j]);
    lower[t]=dot;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(t==0) {
    usable=1;
    for(uint i=0;i<4;++i) for(uint j=0;j<=i;++j) {
      float z=lower[i*4+j];
      for(uint k=0;k<j;++k) z-=lower[i*4+k]*lower[j*4+k];
      if(i==j) {
        if(!(z>1e-5f)) usable=0;
        lower[i*4+j]=sqrt(max(z,1e-5f));
      } else lower[i*4+j]=z/lower[j*4+j];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(usable) {
    float rhs[4]={0,0,0,0};
    float olderr=0;
    for(uint k=0;k<128;++k) {
      float x=float(codes[t*128+k])*(sk[k]*sv[t]/127.0f);
      for(uint i=0;i<4;++i) x+=q[t*4+i]*p[k*4+i];
      float e=s[t*128+k]-x;
      olderr+=e*e;
      for(uint i=0;i<4;++i) rhs[i]+=e*(p[k*4+i]*invnorm[i]);
    }
    for(uint i=0;i<4;++i) {
      for(uint j=0;j<i;++j) rhs[i]-=lower[i*4+j]*rhs[j];
      rhs[i]/=lower[i*4+i];
    }
    for(int i=3;i>=0;--i) {
      for(uint j=uint(i)+1;j<4;++j) rhs[i]-=lower[j*4+uint(i)]*rhs[j];
      rhs[i]/=lower[uint(i)*4+uint(i)];
    }
    float proposed[4];
    for(uint i=0;i<4;++i) proposed[i]=float(half(q[t*4+i]+rhs[i]*invnorm[i]));
    float newerr=0;
    for(uint k=0;k<128;++k) {
      float x=float(codes[t*128+k])*(sk[k]*sv[t]/127.0f);
      for(uint i=0;i<4;++i) x+=proposed[i]*p[k*4+i];
      float e=s[t*128+k]-x;
      newerr+=e*e;
    }
    if(newerr<olderr) for(uint i=0;i<4;++i) factors[t*4+i]=half(proposed[i]);
  }

}


#pragma clang diagnostic pop
