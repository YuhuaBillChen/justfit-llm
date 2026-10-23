// Modified for JustFit: product identity and integration.
#include "model/RuntimeArenas.hpp"
#include <cassert>
#include <iostream>
#include <vector>
using namespace justfit::model;
int main() {
  using T=PrefillTensor;
  std::array<uint64_t,prefillTensorCount> sizes{};
  for(size_t i=0;i<sizes.size();++i) sizes[i]=(i+1)*1024+17;
  const auto plain=prefillArenaLayout(sizes,false), alias=prefillArenaLayout(sizes,true);
  assert(alias.bytes<plain.bytes);
  auto check=[&](const std::vector<T>& live) {
    for (auto a:live) for(auto b:live) if(a!=b) {
      const size_t i=static_cast<size_t>(a),j=static_cast<size_t>(b);
      assert(alias.offsets[i]+sizes[i]<=alias.offsets[j] || alias.offsets[j]+sizes[j]<=alias.offsets[i]);
    }
  };
  std::vector<T> common{T::Hidden0,T::Hidden1,T::Normalized,T::Captured,T::ProjectionSums,T::DownProjectionSums,T::LinearPartials,T::LinearCounters,T::LinearRotated};
  auto phase=[&](std::initializer_list<T> fields){auto live=common;live.insert(live.end(),fields);check(live);};
  phase({T::GdnPacked,T::GdnQueries,T::GdnKeys,T::GdnValues,T::GdnDecay,T::GdnBeta,T::Recurrent,T::GdnHidden,T::GdnOutput});
  phase({T::FullPacked,T::FullQueries,T::ChunkKeys,T::ChunkValues,T::FullAttention,T::AttentionPartials,T::AttentionStatistics,T::AttentionHidden,T::AttentionOutput});
  phase({T::GateIntermediate,T::Intermediate,T::GdnOutput});
  phase({T::GateIntermediate,T::Intermediate,T::AttentionOutput});
  phase({T::TQ4StagedKeys,T::TQ4StagedValues,T::FullPacked,T::FullQueries,T::FullAttention,T::AttentionHidden,T::AttentionOutput});
  phase({T::ContextProjected,T::ContextHidden,T::ContextKv,T::DraftRopeCos,T::DraftRopeSin});
  for(size_t i=0;i<sizes.size();++i)assert(alias.offsets[i]+sizes[i]<=alias.bytes);
  std::cout<<"phase-lifetime nonoverlap PASS, synthetic bytes "<<plain.bytes<<" -> "<<alias.bytes<<"\n";
}
