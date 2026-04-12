#include "render/skybox.h"
#include "render/path_join.h"
#include "render/shader_file.h"

#include <glm/gtc/type_ptr.hpp>

#include <stb_image.h>

#include <cstdio>
#include <string>

#ifndef BP_SKYBOX_ROOT
#define BP_SKYBOX_ROOT "."
#endif

namespace {

constexpr const char kCubemapName[] = "Cubemap_Sky_23-512x512";

static const GLenum kCubeFace[] = {
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
};

static const char* kFaceTag[] = {"posx", "negx", "posy", "negy", "posz", "negz"};

static const float kUnitCube[] = {
    -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f,  -1.f,
    -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,
    1.f,  -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f, 1.f,  1.f,  1.f, 1.f,  1.f,  1.f, 1.f,  -1.f, 1.f,  -1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f, -1.f, 1.f,  1.f, -1.f, 1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
    -1.f, 1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f, 1.f,  1.f,  1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  -1.f,
    -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f, 1.f,  1.f,  1.f, 1.f,  1.f,  1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,
};

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "skybox shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkProgram(GLuint vs, GLuint fs) {
    if (!vs || !fs)
        return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::fprintf(stderr, "skybox program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

} // namespace

bool skyboxInit(SkyboxGL& out, const char* skyboxRootDir) {
    skyboxShutdown(out);

    glGenTextures(1, &out.cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, out.cubemap);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    std::string path;
    bool        ok = true;
    int         w0 = 0, h0 = 0;
    for (int i = 0; i < 6; ++i) {
        char name[256];
        std::snprintf(name, sizeof name, "%s_%s.png", kCubemapName, kFaceTag[i]);
        pathJoin(path, skyboxRootDir, name);
        int w = 0, h = 0;
        unsigned char* data = stbi_load(path.c_str(), &w, &h, nullptr, 4);
        if (!data || w <= 0 || h <= 0) {
            if (data)
                stbi_image_free(data);
            std::fprintf(stderr, "skybox: failed to load face %s\n", path.c_str());
            ok = false;
            break;
        }
        if (i == 0) {
            w0 = w;
            h0 = h;
        } else if (w != w0 || h != h0) {
            std::fprintf(stderr, "skybox: face size mismatch %s\n", path.c_str());
            stbi_image_free(data);
            ok = false;
            break;
        }
        glTexImage2D(kCubeFace[i], 0, GL_SRGB8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        stbi_image_free(data);
    }

    if (!ok) {
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glDeleteTextures(1, &out.cubemap);
        out.cubemap = 0;
        return false;
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    glGenVertexArrays(1, &out.vao);
    glGenBuffers(1, &out.vbo);
    glBindVertexArray(out.vao);
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kUnitCube), kUnitCube, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    std::string vsSrc = loadTextFile(shaderPath("skybox.vert"));
    std::string fsSrc = loadTextFile(shaderPath("skybox.frag"));
    if (vsSrc.empty() || fsSrc.empty()) {
        std::fprintf(stderr, "skybox: missing shaders (tried %s)\n", shaderPath("skybox.vert").c_str());
        skyboxShutdown(out);
        return false;
    }
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());
    out.program = linkProgram(vs, fs);
    if (!out.program) {
        skyboxShutdown(out);
        return false;
    }
    out.locViewProj = glGetUniformLocation(out.program, "uViewProj");
    out.locSky      = glGetUniformLocation(out.program, "uSky");
    return true;
}

void skyboxShutdown(SkyboxGL& sb) {
    if (sb.program)
        glDeleteProgram(sb.program);
    if (sb.vbo)
        glDeleteBuffers(1, &sb.vbo);
    if (sb.vao)
        glDeleteVertexArrays(1, &sb.vao);
    if (sb.cubemap)
        glDeleteTextures(1, &sb.cubemap);
    sb = {};
}

void skyboxDraw(const SkyboxGL& sb, const glm::mat4& proj, const glm::mat4& view) {
    if (!sb.program || !sb.cubemap)
        return;

    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    const glm::mat4 viewRot = glm::mat4(glm::mat3(view));
    const glm::mat4 vp      = proj * viewRot;

    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_FALSE);
    const GLboolean cullWas = glIsEnabled(GL_CULL_FACE);
    glDisable(GL_CULL_FACE);

    glUseProgram(sb.program);
    if (sb.locViewProj >= 0)
        glUniformMatrix4fv(sb.locViewProj, 1, GL_FALSE, glm::value_ptr(vp));
    glActiveTexture(GL_TEXTURE15);
    glBindTexture(GL_TEXTURE_CUBE_MAP, sb.cubemap);
    if (sb.locSky >= 0)
        glUniform1i(sb.locSky, 15);

    glBindVertexArray(sb.vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LESS);
    if (cullWas)
        glEnable(GL_CULL_FACE);
}
