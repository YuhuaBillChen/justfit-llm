#include <metal_stdlib>
using namespace metal;
#include "metal/abi/CompactGDN.h"
static inline uint compact_count(device const uchar *head) {
    return *reinterpret_cast<device const uint *>(head + JUSTFIT_COMPACT_COUNT);
}
constant uint QR = 9216;
constant uint IS=6932,WS=9728,OS=27728;
static inline float product(device const uchar *a){device const float *d=reinterpret_cast<device const float*>(a+27648);float p=1;for(uint j=0;j<16;j++)p=p*d[j];return p;}
static inline float getb(device float *w,uint i){return float(reinterpret_cast<device half*>(w)[i]);}
static inline void setb(device float *w,uint i,float x){reinterpret_cast<device half*>(w)[i]=half(x);}
static inline float bfr(float x){return float(as_type<bfloat>(ushort((as_type<uint>(x)+0x7fffu+((as_type<uint>(x)>>16)&1u))>>16)));}
kernel void lq_flush_all_hoisted(device const float *all [[buffer(0)]],device float *work [[buffer(1)]],device uchar *out [[buffer(2)]],device const float *seed [[buffer(3)]],constant CompactLayerParams &params [[buffer(4)]],uint h [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
 threadgroup float localP[512],localQ[512];
 threadgroup half priorP[512],priorQ[512];
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)out;device const uchar *a=reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS);device float *w=work+h*WS;
 device const char *codes=reinterpret_cast<device const char*>(a);device const float *scales=reinterpret_cast<device const float*>(a+16384);
 device const bfloat *keys=r94_keys(a,h%48);device const float *updates=reinterpret_cast<device const float*>(a+19456),*decay=reinterpret_cast<device const float*>(a+27648);
 for(uint i=t;i<512;i+=128){priorQ[i]=reinterpret_cast<device const half*>(a+17408)[i];priorP[i]=reinterpret_cast<device const half*>(a+17408)[512+i];}
 threadgroup float weights[16];if(t<16){float x=1;for(uint j=t+1;j<16;j++)x=x*decay[j];weights[t]=x;}threadgroup_barrier(mem_flags::mem_threadgroup);
 threadgroup bfloat weightedHi[16*128],weightedLo[16*128];
 for(uint j=0;j<16;j++){float u=updates[j*128+t]*weights[j];float hi=bfr(u),lo=bfr(u-hi);weightedHi[j*128+t]=bfloat(hi);weightedLo[j*128+t]=bfloat(lo);}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float p=product(a);
 for(uint v=0;v<128;v+=4){
  float x0=((float(codes[(v+0)*128+t])*(scales[128+v+0]*(p/127.0f)))*scales[t]);
  float x1=((float(codes[(v+1)*128+t])*(scales[128+v+1]*(p/127.0f)))*scales[t]);
  float x2=((float(codes[(v+2)*128+t])*(scales[128+v+2]*(p/127.0f)))*scales[t]);
  float x3=((float(codes[(v+3)*128+t])*(scales[128+v+3]*(p/127.0f)))*scales[t]);
  for(uint j=0;j<16;j++){const float key=keys[j*128+t];
   x0=x0+float(weightedHi[j*128+v+0])*key;x0=x0+float(weightedLo[j*128+v+0])*key;
   x1=x1+float(weightedHi[j*128+v+1])*key;x1=x1+float(weightedLo[j*128+v+1])*key;
   x2=x2+float(weightedHi[j*128+v+2])*key;x2=x2+float(weightedLo[j*128+v+2])*key;
   x3=x3+float(weightedHi[j*128+v+3])*key;x3=x3+float(weightedLo[j*128+v+3])*key;
  }
  setb(w,(v+0)*128+t,x0);
  setb(w,(v+1)*128+t,x1);
  setb(w,(v+2)*128+t,x2);
  setb(w,(v+3)*128+t,x3);
 }
 device const float *q0=seed;for(uint i=t;i<512;i+=128)localQ[i]=float(half(q0[i]));

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)out;device const uchar *a=reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS);device float *w=work+h*WS;float pn=max(product(a),1e-20f);
 threadgroup float projected[16];
 if(t<16){uint j=t/4,r=t%4;float x=0;for(uint v=0;v<128;v++)x=x+float(priorQ[(v)*4+(j)])*localQ[v*4+r];projected[t]=x;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float x0=0,x1=0,x2=0,x3=0;
 for(uint v=0;v<128;v++){const float b=getb(w,v*128+t);
  x0=x0+b*localQ[v*4];x1=x1+b*localQ[v*4+1];x2=x2+b*localQ[v*4+2];x3=x3+b*localQ[v*4+3];}
 const float accum[4]={x0,x1,x2,x3};
 for(uint r=0;r<4;r++){
  float correction=0;for(uint j=0;j<4;j++)correction=correction+float(priorP[(t)*4+(j)])*projected[j*4+r];
  localP[t*4+r]=float(half(accum[r]+pn*correction));
 }

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)out;device const uchar *a=reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS);device float *w=work+h*WS;float pn=max(product(a),1e-20f);
 threadgroup float projected[16];
 if(t<16){uint j=t/4,r=t%4;float x=0;for(uint k=0;k<128;k++)x=x+float(priorP[(k)*4+(j)])*localP[k*4+r];projected[t]=x;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float x0=0,x1=0,x2=0,x3=0;
 for(uint k=0;k<128;k++){const float b=getb(w,t*128+k);
  x0=x0+b*localP[k*4];x1=x1+b*localP[k*4+1];x2=x2+b*localP[k*4+2];x3=x3+b*localP[k*4+3];}
 const float accum[4]={x0,x1,x2,x3};
 for(uint r=0;r<4;r++){
  float correction=0;for(uint j=0;j<4;j++)correction=correction+float(priorQ[(t)*4+(j)])*projected[j*4+r];
  w[QR+t*4+r]=accum[r]+pn*correction;
 }

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)all;(void)out;device float *w=work+h*WS;float pr[4];for(uint j=0;j<4;j++)pr[j]=w[QR+t*4+j];
 threadgroup float partial[40];
 for(uint pass=0;pass<2;pass++){
  float g[4][4]={},l[4][4]={},inv[4]={};
  for(uint j=0;j<4;j++)for(uint c=0;c<=j;c++){
   float x=pr[j]*pr[c];
   for(uint shift=16;shift>0;shift>>=1)x=x+simd_shuffle_xor(x,shift);
   if(lane==0)partial[simd*10+j*(j+1)/2+c]=x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)for(uint c=0;c<=j;c++){uint i=j*(j+1)/2+c;g[j][c]=((partial[i]+partial[10+i])+partial[20+i])+partial[30+i];}
  for(uint j=0;j<4;j++){
   for(uint c=0;c<j;c++){float x=g[j][c];for(uint z=0;z<c;z++)x=x-l[j][z]*l[c][z];l[j][c]=x*inv[c];}
   float pivot=g[j][j];for(uint z=0;z<j;z++)pivot=pivot-l[j][z]*l[j][z];float floor=j?max(1e-6f*g[j][j],1e-30f):1e-30f;inv[j]=1.0f/sqrt(max(pivot,floor));
  }
  for(uint j=0;j<4;j++){float x=pr[j];for(uint c=0;c<j;c++)x=x-pr[c]*l[j][c];pr[j]=x*inv[j];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint j=0;j<4;j++){w[QR+t*4+j]=pr[j];localQ[t*4+j]=float(half(pr[j]));}

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)out;device const uchar *a=reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS);device float *w=work+h*WS;float pn=max(product(a),1e-20f);
 threadgroup float projected[16];
 if(t<16){uint j=t/4,r=t%4;float x=0;for(uint v=0;v<128;v++)x=x+float(priorQ[(v)*4+(j)])*localQ[v*4+r];projected[t]=x;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float x0=0,x1=0,x2=0,x3=0;
 for(uint v=0;v<128;v++){const float b=getb(w,v*128+t);
  x0=x0+b*localQ[v*4];x1=x1+b*localQ[v*4+1];x2=x2+b*localQ[v*4+2];x3=x3+b*localQ[v*4+3];}
 const float accum[4]={x0,x1,x2,x3};
 for(uint r=0;r<4;r++){
  float correction=0;for(uint j=0;j<4;j++)correction=correction+float(priorP[(t)*4+(j)])*projected[j*4+r];
  localP[t*4+r]=float(half(accum[r]+pn*correction));
 }

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)out;device const uchar *a=reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS);device float *w=work+h*WS;float pn=max(product(a),1e-20f);
 for(uint i=t;i<16384;i+=128){uint v=i/128,k=i%128;float old=0,newrank=0;
  for(uint r=0;r<4;r++){old=old+float(priorQ[(v)*4+(r)])*float(priorP[(k)*4+(r)]);newrank=newrank+localQ[v*4+r]*localP[k*4+r];}
  setb(w,i,(getb(w,i)/pn+old)*pn-newrank);
 }

}
 threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
{

 if(params.only_full && *reinterpret_cast<device const uint*>(reinterpret_cast<device const uchar*>(all+(h/48)*352256+(h%48)*IS)+27712)!=16)return;
 (void)seed;

 (void)all;device float *w=work+h*WS;device uchar *o=out+(h/48)*1409024+(h%48)*OS;device float *scales=reinterpret_cast<device float*>(o+16384);device half *factors=reinterpret_cast<device half*>(o+17408);
 float mean=0;for(uint v=0;v<128;v++)mean=mean+abs(getb(w,v*128+t));scales[t]=max(sqrt(mean/128),1e-12f);
 threadgroup_barrier(mem_flags::mem_device);
 float largest=0;for(uint k=0;k<128;k++)largest=max(largest,abs(getb(w,t*128+k)/scales[k]));scales[128+t]=max(largest,1e-12f);
 for(uint k=0;k<128;k++)reinterpret_cast<device char*>(o)[t*128+k]=char(clamp(rint((getb(w,t*128+k)/scales[k])*(127.0f/scales[128+t])),-127.0f,127.0f));
 for(uint r=0;r<4;r++){factors[t*4+r]=half(localQ[t*4+r]);factors[512+t*4+r]=half(localP[t*4+r]);}
 threadgroup_barrier(mem_flags::mem_device);
 if(t==0)*reinterpret_cast<device uint*>(o+27712)=0;

}
}

// Identity-pad a committed partial window before the unchanged 16-record codec.
kernel void lq_flush_prepare_early(device float *all [[buffer(0)]],device float *work [[buffer(1)]],device uchar *out [[buffer(2)]],device const float *seed [[buffer(3)]],constant CompactLayerParams &params [[buffer(4)]],uint h [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]]){
 (void)work;(void)out;(void)seed;(void)params;
 device uchar *a=reinterpret_cast<device uchar*>(all+(h/48)*352256+(h%48)*IS);
 const uint count=compact_count(a);if(count<=8 || count>=16)return;
 device bfloat *keys=r94_keys(a,h%48);device float *updates=reinterpret_cast<device float*>(a+19456),*decay=reinterpret_cast<device float*>(a+27648);
 for(uint j=count;j<16;j++){if(h%3==0)keys[j*128+t]=bfloat(0);updates[j*128+t]=0;if(t==0)decay[j]=1;}
 threadgroup_barrier(mem_flags::mem_device);
 if(t==0)*reinterpret_cast<device uint*>(a+27712)=16;
}
