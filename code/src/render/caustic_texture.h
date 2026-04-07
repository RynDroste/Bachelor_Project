#pragma once

#include <glad/glad.h>

// sRGB 2D texture, repeat + mips. Returns 0 if path missing/invalid.
GLuint loadCausticTexture(const char* path);
