#pragma once

#include <string>

// Join BP_SHADER_ROOT (absolute, set by CMake) with a filename under code/shaders/.
std::string shaderPath(const char* filename);

std::string loadTextFile(const std::string& path);
