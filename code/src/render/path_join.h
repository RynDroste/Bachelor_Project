#pragma once

#include <string>

inline void pathJoin(std::string& out, const char* dir, const char* file) {
    out.assign(dir ? dir : "");
    if (!out.empty()) {
        const char c = out.back();
        if (c != '/' && c != '\\')
            out.push_back('/');
    }
    out += file;
}
