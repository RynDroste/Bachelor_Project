#pragma once

#include <glm/glm.hpp>

// RGB16F cubemap + full mip chain; radiance aligned with world +Y up.
// lightDirTowardSun: same as shader uLightDir (direction from surface toward sun).
unsigned int makeProceduralEnvCubemap(int facePixels, const glm::vec3& lightDirTowardSun, float* outMaxMipLevel);
