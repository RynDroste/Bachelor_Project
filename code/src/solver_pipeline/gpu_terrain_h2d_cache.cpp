#include "solver_pipeline/gpu_terrain_h2d_cache.hpp"

#include <cstring>
#include <vector>

namespace bp_gpu {
namespace {

std::vector<float>       s_shadow;
bool                     s_sweDterrainMatchesShadow = false;

void saveShadow(const float* terrain, std::size_t nfloats) {
    if (!terrain || nfloats == 0)
        return;
    s_shadow.resize(nfloats);
    std::memcpy(s_shadow.data(), terrain, nfloats * sizeof(float));
}

} // namespace

bool terrainHostMatchesCachedSnapshot(const float* terrain, std::size_t nfloats) {
    if (!terrain || nfloats == 0)
        return false;
    if (s_shadow.size() != nfloats)
        return false;
    return std::memcmp(s_shadow.data(), terrain, nfloats * sizeof(float)) == 0;
}

void noteWaveDecomposeTerrainH2d(const float* terrain, std::size_t nfloats) {
    saveShadow(terrain, nfloats);
    s_sweDterrainMatchesShadow = false;
}

void noteSweTerrainH2d(const float* terrain, std::size_t nfloats) {
    saveShadow(terrain, nfloats);
    s_sweDterrainMatchesShadow = true;
}

bool sweTerrainDeviceMatchesHostCache(const float* terrain, std::size_t nfloats) {
    return s_sweDterrainMatchesShadow && terrainHostMatchesCachedSnapshot(terrain, nfloats);
}

void terrainCacheInvalidate() {
    s_shadow.clear();
    s_sweDterrainMatchesShadow = false;
}

} // namespace bp_gpu
