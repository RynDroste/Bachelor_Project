#pragma once

#include <glad/glad.h>

struct TerrainSand04Textures {
    GLuint albedo   = 0;
    GLuint normalGl = 0;
    GLuint ao       = 0;
    GLuint roughness = 0;
};

bool loadTerrainSand04(const char* dir, TerrainSand04Textures& out);
void destroyTerrainSand04(TerrainSand04Textures& t);
GLuint loadCausticTexture(const char* path);
