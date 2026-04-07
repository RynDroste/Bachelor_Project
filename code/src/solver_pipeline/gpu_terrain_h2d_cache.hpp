#pragma once

#include <cstddef>

namespace bp_gpu {

bool terrainHostMatchesCachedSnapshot(const float* terrain, std::size_t nfloats);

void noteWaveDecomposeTerrainH2d(const float* terrain, std::size_t nfloats);

void noteSweTerrainH2d(const float* terrain, std::size_t nfloats);

bool sweTerrainDeviceMatchesHostCache(const float* terrain, std::size_t nfloats);

void terrainCacheInvalidate();

} // namespace bp_gpu
