// Modified for JustFit: product identity and integration.
#pragma once
#include <cstdlib>
#include <string_view>
namespace justfit::model {
inline bool hostEmbeddingEnabled() noexcept { const char *v=std::getenv("JUSTFIT_HOST_EMBEDDING"); return v && std::string_view(v)=="1"; }
inline bool elasticPrefillArenaEnabled() noexcept { const char *v=std::getenv("JUSTFIT_ELASTIC_PREFILL"); return v && std::string_view(v)=="1"; }
}
