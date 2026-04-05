// Tracks the last bathymetry (B) copied from host to GPU so we can skip redundant H2D.
// Wave-decompose (d_b) and SWE (d_terrain) use separate device allocations; SWE may only
// skip when its d_terrain already matches the cached host snapshot.

#pragma once

#include <cstddef>

namespace bp_gpu {

bool terrainHostMatchesCachedSnapshot(const float* terrain, std::size_t nfloats);

// Call after Host->Device copy into wave-decompose d_b.
void noteWaveDecomposeTerrainH2d(const float* terrain, std::size_t nfloats);

// Call after Host->Device copy into SWE d_terrain.
void noteSweTerrainH2d(const float* terrain, std::size_t nfloats);

// True iff SWE d_terrain was filled for the same bathymetry bytes as `terrain`.
bool sweTerrainDeviceMatchesHostCache(const float* terrain, std::size_t nfloats);

void terrainCacheInvalidate();

} // namespace bp_gpu
