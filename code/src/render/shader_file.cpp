#include "render/shader_file.h"
#include "render/path_join.h"

#include <fstream>
#include <sstream>

#ifndef BP_SHADER_ROOT
#define BP_SHADER_ROOT "shaders"
#endif

std::string shaderPath(const char* filename) {
    std::string out;
    pathJoin(out, BP_SHADER_ROOT, filename);
    return out;
}

std::string loadTextFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
