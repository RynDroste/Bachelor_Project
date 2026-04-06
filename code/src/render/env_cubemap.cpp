#include "render/env_cubemap.h"

#include <glad/glad.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace {

void appendPath(std::string& out, const char* dir, const char* name) {
    out.assign(dir);
    if (!out.empty()) {
        const char c = out.back();
        if (c != '/' && c != '\\')
            out.push_back('/');
    }
    out += name;
}

glm::vec3 cubemapDirection(unsigned face, float u, float v) {
    u = u * 2.f - 1.f;
    v = v * 2.f - 1.f;
    glm::vec3 dir(0.f);
    switch (face) {
    case 0:
        dir = glm::vec3(1.f, -v, -u);
        break; // +X
    case 1:
        dir = glm::vec3(-1.f, -v, u);
        break; // -X
    case 2:
        dir = glm::vec3(u, 1.f, v);
        break; // +Y
    case 3:
        dir = glm::vec3(u, -1.f, -v);
        break; // -Y
    case 4:
        dir = glm::vec3(u, -v, 1.f);
        break; // +Z
    case 5:
        dir = glm::vec3(-u, -v, -1.f);
        break; // -Z
    default:
        dir = glm::vec3(0.f, 1.f, 0.f);
        break;
    }
    return glm::normalize(dir);
}

glm::vec3 sampleSky(const glm::vec3& w, const glm::vec3& lightTowardSun) {
    const glm::vec3 sunDir = glm::normalize(lightTowardSun);
    const float   y        = glm::clamp(w.y, -1.f, 1.f);
    glm::vec3     zenith(0.22f, 0.45f, 0.92f);
    glm::vec3     horizon(0.58f, 0.70f, 0.94f);
    float         skyT = std::pow(std::max(0.f, y * 0.55f + 0.45f), 0.82f);
    glm::vec3     sky  = glm::mix(horizon, zenith, skyT);
    const float   cosSun = glm::dot(w, sunDir);
    const float   edge   = std::cos(1.2f * 3.14159265f / 180.f);
    const float   sunDisk = glm::smoothstep(edge, 1.f, cosSun);
    const glm::vec3 sunCol(2.6f, 2.35f, 2.0f);
    return sky + sunCol * sunDisk;
}

unsigned int makeProceduralEnvCubemap(int facePixels, const glm::vec3& lightDirTowardSun, float* outMaxMipLevel) {
    if (facePixels < 4)
        facePixels = 4;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    std::vector<float> buf(static_cast<std::size_t>(facePixels) * facePixels * 3u);

    for (unsigned face = 0; face < 6; ++face) {
        for (int j = 0; j < facePixels; ++j) {
            for (int i = 0; i < facePixels; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / static_cast<float>(facePixels);
                const float v = (static_cast<float>(j) + 0.5f) / static_cast<float>(facePixels);
                const glm::vec3 dir = cubemapDirection(face, u, v);
                const glm::vec3 c   = sampleSky(dir, lightDirTowardSun);
                const std::size_t o =
                    (static_cast<std::size_t>(j) * static_cast<std::size_t>(facePixels) + static_cast<std::size_t>(i)) * 3u;
                buf[o + 0] = c.r;
                buf[o + 1] = c.g;
                buf[o + 2] = c.b;
            }
        }
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGB16F, facePixels, facePixels, 0, GL_RGB, GL_FLOAT,
                     buf.data());
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    const int nmip = 1 + static_cast<int>(std::floor(std::log2(static_cast<float>(facePixels))));
    if (outMaxMipLevel)
        *outMaxMipLevel = static_cast<float>(nmip - 1);
    return static_cast<unsigned int>(tex);
}

unsigned int tryLoadSkyboxJpegs(const char* skyboxDir, float* outMaxMipLevel) {
    static const char* kFaceFiles[] = {"right.jpg", "left.jpg", "top.jpg", "bottom.jpg", "front.jpg", "back.jpg"};

    int w0 = 0, h0 = 0;
    stbi_set_flip_vertically_on_load(0);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::string path;
    for (unsigned face = 0; face < 6; ++face) {
        appendPath(path, skyboxDir, kFaceFiles[face]);
        int w = 0, h = 0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, nullptr, 3);
        if (!data || w <= 0 || h <= 0 || w != h) {
            if (data)
                stbi_image_free(data);
            std::fprintf(stderr, "env cubemap: failed to load %s\n", path.c_str());
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

unsigned int createEnvCubemap(const char* skyboxDir, const glm::vec3& lightDirTowardSun, float* outMaxMipLevel) {
    if (skyboxDir && skyboxDir[0] != '\0') {
        const unsigned t = tryLoadSkyboxJpegs(skyboxDir, outMaxMipLevel);
        if (t)
            return t;
        std::fprintf(stderr, "warning: skybox folder load failed; using procedural sky\n");
    }
    return makeProceduralEnvCubemap(256, lightDirTowardSun, outMaxMipLevel);
}
