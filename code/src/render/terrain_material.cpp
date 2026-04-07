#include "render/terrain_material.h"
#include "render/path_join.h"

#include <stb_image.h>

#include <cstdio>
#include <string>

namespace {

void tex2DRepeatMipFinish() {
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
}

GLuint makeFallbackR8(unsigned char v) {
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 1, 1, 0, GL_RED, GL_UNSIGNED_BYTE, &v);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    return t;
}

GLuint makeFallbackRGBA(unsigned char r, unsigned char g, unsigned char b, bool srgb) {
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    const unsigned char px[4] = {r, g, b, 255};
    const GLenum internal = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    glTexImage2D(GL_TEXTURE_2D, 0, internal, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

GLuint upload2DRGBA(const char* path, bool srgb, const char* label) {
    int w = 0, h = 0;
    unsigned char* data = stbi_load(path, &w, &h, nullptr, 4);
    if (!data || w <= 0 || h <= 0) {
        if (data)
            stbi_image_free(data);
        std::fprintf(stderr, "terrain material: missing %s (%s)\n", label, path);
        return 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    const GLenum internal = srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    tex2DRepeatMipFinish();
    return tex;
}

GLuint upload2DR8(const char* path, const char* label) {
    int w = 0, h = 0;
    unsigned char* data = stbi_load(path, &w, &h, nullptr, 1);
    if (!data || w <= 0 || h <= 0) {
        if (data)
            stbi_image_free(data);
        std::fprintf(stderr, "terrain material: missing %s (%s)\n", label, path);
        return 0;
    }
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
    tex2DRepeatMipFinish();
    return tex;
}

} // namespace

bool loadTerrainSand04(const char* dir, TerrainSand04Textures& out) {
    destroyTerrainSand04(out);

    std::string p;
    pathJoin(p, dir, "sand_01_color_1k.png");
    out.albedo = upload2DRGBA(p.c_str(), true, "albedo");
    if (!out.albedo)
        out.albedo = makeFallbackRGBA(255, 255, 255, true);

    pathJoin(p, dir, "sand_01_normal_gl_1k.png");
    out.normalGl = upload2DRGBA(p.c_str(), false, "normal");
    if (!out.normalGl)
        out.normalGl = makeFallbackRGBA(128, 128, 255, false);

    pathJoin(p, dir, "sand_01_ambient_occlusion_1k.png");
    out.ao = upload2DR8(p.c_str(), "AO");
    if (!out.ao)
        out.ao = makeFallbackR8(255);

    pathJoin(p, dir, "sand_01_roughness_1k.png");
    out.roughness = upload2DR8(p.c_str(), "roughness");
    if (!out.roughness)
        out.roughness = makeFallbackR8(128);

    return true;
}

void destroyTerrainSand04(TerrainSand04Textures& t) {
    const GLuint ids[] = {t.albedo, t.normalGl, t.ao, t.roughness};
    for (GLuint id : ids) {
        if (id)
            glDeleteTextures(1, &id);
    }
    t = {};
}

GLuint loadCausticTexture(const char* path) {
    GLuint tex = upload2DRGBA(path, false, "caustic");
    if (!tex)
        std::fprintf(stderr, "caustic texture not found: %s\n", path);
    return tex;
}
