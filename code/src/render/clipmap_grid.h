#pragma once

// Geo-clipmap for a large flat-footprint surface (water / terrain).
// L concentric squares share a quad grid of step `d_l = baseSpacing * 2^l`:
//   - level 0: a solid (2N x 2N) quad mesh covering 2N*d_0 on a side
//   - level l>0: the same (2N x 2N) mesh with the central (N x N) block removed
//     (a "hollow ring"), covering 2N*d_l on a side with an N*d_l hole that
//     exactly contains the next-finer level's outer footprint.
//
// Each level is independently snapped to the camera XZ on a stride of 2*d_l to
// avoid per-vertex crawling. The vertex shader receives (aLocalXZ, ringCenter,
// ringSpacing) and reconstructs worldXZ = ringCenter + aLocalXZ * ringSpacing.

#include <glad/glad.h>
#include <glm/vec2.hpp>

struct ClipmapGrid {
    int    N            = 0;    // each level spans [-N, N] along each axis (2N quads per side)
    int    L            = 0;    // number of levels, >=1
    float  baseSpacing  = 0.f;  // d_0

    GLuint solidVao     = 0;
    GLuint solidVbo     = 0;
    GLuint solidIbo     = 0;
    GLsizei solidCount  = 0;

    GLuint ringVao      = 0;
    GLuint ringVbo      = 0;
    GLuint ringIbo      = 0;
    GLsizei ringCount   = 0;
};

// Build shared solid / ring VAOs. Returns false on GL failure.
// N must be even (ring hole is N/2 quads deep on each side).
bool clipmapGridInit(ClipmapGrid& g, int N, int L, float baseSpacing);

void clipmapGridShutdown(ClipmapGrid& g);

// Snap a level's world-space center to the camera XZ on a 2*spacing grid.
glm::vec2 clipmapLevelCenter(const glm::vec2& cameraXZ, float levelSpacing);

// Shared center used for EVERY level, so that ring boundaries between adjacent
// levels line up exactly (no geometric seams). Quantizes the camera XZ to the
// snap-step of the outermost (coarsest) level; with that choice the shared
// center is automatically a valid snap point for every finer level too.
glm::vec2 clipmapSharedCenter(const glm::vec2& cameraXZ, const ClipmapGrid& g);

// Returns d_l for this level.
float clipmapLevelSpacing(const ClipmapGrid& g, int level);

// Binds the VAO for this level and issues the draw call.
// Level 0 uses the solid mesh; level >= 1 uses the ring mesh.
void clipmapGridDrawLevel(const ClipmapGrid& g, int level);
