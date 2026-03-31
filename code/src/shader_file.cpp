#include "shader_file.h"

#include <fstream>
#include <sstream>

#ifndef BP_SHADER_ROOT
#define BP_SHADER_ROOT "shaders"
#endif

std::string shaderPath(const char* filename) {
    std::string root(BP_SHADER_ROOT);
    if (!root.empty()) {
        const char c = root.back();
        if (c != '/' && c != '\\')
            root += '/';
    }
    return root + filename;
}

std::string loadTextFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
