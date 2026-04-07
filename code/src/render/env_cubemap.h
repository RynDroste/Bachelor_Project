#pragma once

#include <glm/glm.hpp>

unsigned int createEnvCubemap(const char* skyboxDir, const glm::vec3& lightDirTowardSun, float* outMaxMipLevel);
