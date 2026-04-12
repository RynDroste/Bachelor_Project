#include "render/env_cubemap.h"
#include "render/path_join.h"

#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace {

unsigned int tryLoadSkyboxFaces(const char* skyboxDir, float* outMaxMipLevel) {
    static const char* kFaceBases[] = {"right", "left", "top", "bottom", "front", "back"};
    static const char* kExts[]    = {".png", ".jpg"};

    int w0 = 0, h0 = 0;
    stbi_set_flip_vertically_on_load(0);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::string path;
    std::string name;
    for (unsigned face = 0; face < 6; ++face) {
        unsigned char* data = nullptr;
        int            w = 0, h = 0;
        for (const char* ext : kExts) {
            name.assign(kFaceBases[face]);
            name += ext;
            pathJoin(path, skyboxDir, name.c_str());
            data = stbi_load(path.c_str(), &w, &h, nullptr, 3);
            if (data && w > 0 && h > 0 && w == h)
                break;
            if (data) {
                stbi_image_free(data);
                data = nullptr;
            }
        }
        if (!data) {
            std::fprintf(stderr, "env cubemap: failed to load %s (%s.*)\n", skyboxDir ? skyboxDir : "", kFaceBases[face]);
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
            glDeleteTextures(1, &tex);
            return 0;
        }
        if (face == 0) {
            w0 = w;
            h0 = h;
        } else if (w != w0 || h != h0) {
            stbi_image_free(data);
            std::fprintf(stderr, "env cubemap: face size mismatch in %s\n", path.c_str());
            glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
            glDeleteTextures(1, &tex);
            return 0;
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_SRGB8, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    const int nmip = 1 + static_cast<int>(std::floor(std::log2(static_cast<float>(w0))));
    if (outMaxMipLevel)
        *outMaxMipLevel = static_cast<float>(nmip - 1);
    return static_cast<unsigned int>(tex);
}

} // namespace

unsigned int createEnvCubemap(const char* skyboxDir, float* outMaxMipLevel) {
    if (!skyboxDir || skyboxDir[0] == '\0')
        return 0;
    return tryLoadSkyboxFaces(skyboxDir, outMaxMipLevel);
}
