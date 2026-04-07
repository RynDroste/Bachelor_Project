#pragma once

#include <glad/glad.h>

struct TerrainSand04Textures {
    GLuint albedo   = 0;
    GLuint normalGl = 0;
    GLuint ao       = 0;
    GLuint roughness = 0;
};

// Loads sand_01_*_1k.png from dir (trailing slash optional). Others fall back to 1x1 if missing.
bool loadTerrainSand04(const char* dir, TerrainSand04Textures& out);

void destroyTerrainSand04(TerrainSand04Textures& t);

// Loads a single caustic texture (RGBA, repeating, mipmapped). Returns 0 on failure.
GLuint loadCausticTexture(const char* path);
