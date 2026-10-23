// Modified for JustFit: product identity and integration.
#include "ops/Sampling.hpp"
#include "ops/VerifyShape.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace justfit;
int main(int argc,char **argv){
 assert(argc==2);metal::MetalBackend backend(argv[1]);
 auto alloc=[&](uint64_t n){auto b=backend.allocateBuffer(n,metal::BufferStorage::Shared,"evo4-sampling-test");std::memset(b.contents(),0,n);return b;};
 constexpr uint32_t V=128,B=2,C=JUSTFIT_DRAFT_CANDIDATES;
 for(uint32_t rows:{8u,16u}){
  ops::VerifyShapeScope shape(rows);const auto w=ops::Sampling::workspace(B*rows);const uint32_t P=rows-1;
  ops::SamplingBuffers b{alloc(B*rows*V*4),alloc(w.partialMassesBytes),alloc(w.vocabularyRowsBytes),alloc(B*JUSTFIT_SAMPLING_UNIFORMS*4),alloc(B*(rows+1)*4),alloc(B*rows*4),alloc(w.argmaxValuesBytes),alloc(w.argmaxIndicesBytes),alloc(B*rows*4),alloc(B*P*C*4),alloc(B*P*C*4),alloc(w.vocabularyRangesBytes),alloc(w.vocabularyArrivalsBytes)};
  auto proposals=alloc(B*P*4),retained=alloc(B*4),accepted=alloc(B*4);
  ops::Sampling sampler(V);std::array<ops::SamplingPolicy,B> policies{};for(auto&p:policies)p.temperature=1;
  auto *logits=static_cast<float*>(b.logits.contents());
  auto *ids=static_cast<uint32_t*>(b.draftCandidates.contents());auto *prob=static_cast<float*>(b.draftProbabilities.contents());auto *u=static_cast<float*>(b.uniforms.contents());
  for(uint32_t row=0;row<B*rows;++row){std::fill_n(logits+row*V,V,-INFINITY);logits[row*V]=std::log(.25f);logits[row*V+1]=std::log(.75f);}
  std::fill_n(ids,B*P*C,UINT32_MAX);for(uint32_t p=0;p<B*P;++p){ids[p*C]=0;prob[p*C]=1;}
  for(uint32_t reject:{0u,6u,7u,14u,15u}){
   if(reject>P)continue;
   std::fill_n(u,B*JUSTFIT_SAMPLING_UNIFORMS,.1f);
   // Lane0 rejects at requested position; lane1 accepts the entire draft.
   if(reject<P)u[reject<7?JUSTFIT_UNIFORM_ACCEPTANCE+reject:16+reject-7]=.5f;
   std::array<uint32_t,B> limits{rows,rows};metal::CommandGraph graph;
   sampler.addVerify(graph,policies,b,126,127,{});
   sampler.addAcceptance(graph,{proposals,b.draftCandidates,b.draftProbabilities,b.vocabularyRows,b.uniforms,b.outputTokens,retained,accepted},limits,policies,126,127);
   (void)backend.submitCommandAsync(graph.command()).wait();
   auto *a=static_cast<uint32_t*>(accepted.contents()),*r=static_cast<uint32_t*>(retained.contents()),*out=static_cast<uint32_t*>(b.outputTokens.contents());
   assert(a[0]==reject && r[0]==reject+1);assert(a[1]==P && r[1]==rows);
   if(reject<P)assert(out[reject]==1); // Exact residual of p=(.25,.75), q=(1,0).
   for(uint32_t i=0;i<reject;++i)assert(out[i]==0);
   auto *p=static_cast<TargetVocabularyRow*>(b.vocabularyRows.contents());
   for(uint32_t lane=0;lane<B;++lane)for(uint32_t i=0;i<P;++i)assert(std::abs(p[lane*rows+i].draft_probability-.25f)<1e-5f);
  }
  std::cout<<"PASS rows="<<rows<<" B2 exact residual, tail rejection, acceptance stride\n";
 }
}
