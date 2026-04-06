#pragma once

#include <glm/glm.hpp>

// Loads six JPG faces from skyboxDir: right/left/top/bottom/front/back.jpg (+Y up, matches GL cubemap faces).
// RGB8/SRGB cubemap + full mip chain. On load failure, falls back to procedural RGB16F sky (same as before).
unsigned int createEnvCubemap(const char* skyboxDir, const glm::vec3& lightDirTowardSun, float* outMaxMipLevel);
