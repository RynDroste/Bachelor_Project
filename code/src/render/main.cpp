#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "solver_pipeline/airy_fftw.h"
#include "solver_pipeline/pipeline.h"
#include "render/boat.h"
#include "render/env_cubemap.h"
#include "render/terrain_material.h"
#include "render/shader_file.h"
#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/wavedecomposer.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifndef BP_TOOLS_ROOT
#define BP_TOOLS_ROOT "."
#endif
#ifndef BP_SKYBOX_ROOT
#define BP_SKYBOX_ROOT "."
#endif
#ifndef BP_SAND01_ROOT
#define BP_SAND01_ROOT "."
#endif

namespace {

constexpr int   kNx = 256;
constexpr int   kNy = 256;
constexpr float kDx = 1.0f;
constexpr float kDt = 1.0f / 120.0f;
constexpr int   kSubsteps = 2;
constexpr bool  kCoupledStep = true;
constexpr bool  kSplitCompareSwe = false;
constexpr float kGradPenaltyD = 0.25f;
constexpr bool  kVsync          = false;
constexpr int   kWindowWidth    = 1280;
constexpr int   kWindowHeight   = 720;
constexpr int   kWaveDiffuseIters = 8; 
constexpr float kCamFovDeg      = 55.f;
constexpr float kCamTargetY     = 3.5f;
constexpr glm::vec3 kFixedCamEye(70.956f, 44.f, 81.118f);
constexpr glm::vec3 kFixedCamTarget(0.f, kCamTargetY, 0.f);
constexpr float kReflectPlaneY = 4.0f;
constexpr float kEtaRef = kReflectPlaneY;
constexpr float kWetDepthEps = 1e-3f;
constexpr float kShoreBlendRange = 2.0f;
constexpr float kGerstnerWeight = 1.0f;
constexpr float kWaterWaveScale      = 0.09f;
constexpr float kWaterWaveStrength   = 0.055f;
constexpr float kWaterAnimationSpeed = 0.85f;
constexpr float kWaterIOR                = 1.33f;
constexpr float kWaterRefractionParallax = 1.0f;
constexpr float kWaterReflections        = 0.86f;
constexpr glm::vec3 kWaterColor(0.93f, 0.97f, 1.0f);
constexpr float kWaterAlpha              = 0.76f;
constexpr float kClipNear      = 0.1f;
constexpr float kClipFar       = 500.f;
constexpr float kDepthAbsorb   = 0.008f;
constexpr float kRefractionMaxOffset = 0.06f;
constexpr float kRefractionLinTol    = 0.12f;
constexpr float kEnvWaveRough = 0.14f;
constexpr bool kRenderTerrainMesh = true;
constexpr float kTerrainMaterialUvScale = 0.035f;

// Skybox: unit cube, 36 verts, interior view.
static const float kSkyboxVerts[] = {
    -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,  -1.f,
    -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,
    1.f,  -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f, 1.f,  1.f,  1.f, 1.f,  1.f,  1.f, 1.f,  -1.f, 1.f,  -1.f, -1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f, -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, 1.f,
    -1.f, 1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  1.f,  1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  -1.f,
    -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f, -1.f, 1.f,  1.f, -1.f, 1.f,  1.f, -1.f, -1.f, -1.f, -1.f, -1.f,
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
        std::fprintf(stderr, "shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint makeProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) {
        if (v) glDeleteShader(v);
        if (f) glDeleteShader(f);
        return 0;
    }
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::fprintf(stderr, "program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

bool loadTerrainHeightmapRaw(const char* path, Grid& g) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    const size_t n = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    in.read(reinterpret_cast<char*>(g.terrain.data()), static_cast<std::streamsize>(n * sizeof(float)));
    return static_cast<size_t>(in.gcount()) == n * sizeof(float);
}

void buildWaterCornerIJ(int nx, int ny, std::vector<float>& out) {
    const int vx = nx + 1;
    const int vz = ny + 1;
    out.resize(static_cast<size_t>(vx * vz * 2));
    size_t k = 0;
    for (int j = 0; j < vz; ++j) {
        for (int i = 0; i < vx; ++i) {
            out[k++] = static_cast<float>(i);
            out[k++] = static_cast<float>(j);
        }
    }
}

void allocGridTextures(int nx, int ny, GLuint& texH, GLuint& texB) {
    glGenTextures(1, &texH);
    glGenTextures(1, &texB);
    for (int pass = 0; pass < 2; ++pass) {
        const GLuint t = (pass == 0) ? texH : texB;
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, nullptr);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void uploadTerrainTexture(const Grid& g, GLuint texB) {
    glBindTexture(GL_TEXTURE_2D, texB);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.NX, g.NY, GL_RED, GL_FLOAT, g.terrain.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void uploadWaterDepthTexture(const Grid& g, GLuint texH) {
    glBindTexture(GL_TEXTURE_2D, texH);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.NX, g.NY, GL_RED, GL_FLOAT, g.h.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void buildGridIndices(int nx, int ny, std::vector<unsigned>& idx) {
    const int vx = nx + 1;
    idx.clear();
    idx.reserve(static_cast<size_t>(nx * ny * 6));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            unsigned i0 = static_cast<unsigned>(i + j * vx);
            unsigned i1 = i0 + 1u;
            unsigned i2 = static_cast<unsigned>(i + (j + 1) * vx);
            unsigned i3 = i2 + 1u;
            idx.push_back(i0);
            idx.push_back(i2);
            idx.push_back(i1);
            idx.push_back(i1);
            idx.push_back(i2);
            idx.push_back(i3);
        }
    }
}

struct FrameCtx {
    int fbW = 800;
    int fbH = 600;
};

glm::mat4 reflectionViewAcrossY(float planeY, const glm::vec3& eye, const glm::vec3& target) {
    auto rfl = [planeY](glm::vec3 p) {
        p.y = 2.f * planeY - p.y;
        return p;
    };
    return glm::lookAt(rfl(eye), rfl(target), glm::vec3(0.f, -1.f, 0.f));
}

void ensureReflectionFBO(int w, int h, GLuint& fbo, GLuint& colorTex, GLuint& depthRbo, int& curW, int& curH) {
    if (w <= 0 || h <= 0)
        return;
    if (fbo && w == curW && h == curH)
        return;
    if (fbo) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &colorTex);
        glDeleteRenderbuffers(1, &depthRbo);
        fbo = 0;
        colorTex = 0;
        depthRbo = 0;
    }
    curW = w;
    curH = h;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &colorTex);
    glGenRenderbuffers(1, &depthRbo);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRbo);
    const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (st != GL_FRAMEBUFFER_COMPLETE)
        std::fprintf(stderr, "reflection FBO incomplete (status 0x%x)\n", static_cast<unsigned>(st));
}

void ensureRefractTex(int w, int h, GLuint& tex, int& curW, int& curH) {
    if (w <= 0 || h <= 0)
        return;
    if (tex && w == curW && h == curH)
        return;
    if (tex) {
        glDeleteTextures(1, &tex);
        tex = 0;
    }
    curW = w;
    curH = h;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void ensureRefractDepthFbo(int w, int h, GLuint& fbo, GLuint& depthTex, int& curW, int& curH) {
    if (w <= 0 || h <= 0)
        return;
    if (fbo && w == curW && h == curH)
        return;
    if (fbo) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &depthTex);
        fbo = 0;
        depthTex = 0;
    }
    curW = w;
    curH = h;
    glGenFramebuffers(1, &fbo);
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, w, h, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "refract depth FBO incomplete (status 0x%x)\n", static_cast<unsigned>(st));
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &depthTex);
        fbo      = 0;
        depthTex = 0;
        curW = curH = 0;
    }
}

void framebufferSizeCB(GLFWwindow* w, int width, int height) {
    auto* ctx = static_cast<FrameCtx*>(glfwGetWindowUserPointer(w));
    if (ctx) {
        ctx->fbW = width;
        ctx->fbH = height;
    }
    glViewport(0, 0, width, height);
}

void fillBoatSolidMesh(std::vector<float>& out, const Boat& boat) {
    out.clear();
    out.reserve(6u * 3u * 12u);

    constexpr float kVisDraft     = 0.28f;
    constexpr float kVisFreeboard = 1.05f;

    const float co = std::cos(boat.heading);
    const float si = std::sin(boat.heading);
    const float bx = boat.pos.x;
    const float bz = boat.pos.y;
    const float hL = boat.length * 0.5f;
    const float hW = boat.width * 0.5f;
    const float yBottom = boat.z - kVisDraft;
    const float yTop = boat.z + kVisFreeboard;

    auto P = [&](float u, float v, float y) {
        return glm::vec3(bx + co * u - si * v, y, bz + si * u + co * v);
    };

    const glm::vec3 v0 = P(-hL, -hW, yBottom);
    const glm::vec3 v1 = P(+hL, -hW, yBottom);
    const glm::vec3 v2 = P(+hL, +hW, yBottom);
    const glm::vec3 v3 = P(-hL, +hW, yBottom);
    const glm::vec3 v4 = P(-hL, -hW, yTop);
    const glm::vec3 v5 = P(+hL, -hW, yTop);
    const glm::vec3 v6 = P(+hL, +hW, yTop);
    const glm::vec3 v7 = P(-hL, +hW, yTop);

    const glm::vec3 nBow = glm::normalize(glm::vec3(co, 0.f, si));
    const glm::vec3 nStern = -nBow;
    const glm::vec3 nStar = glm::normalize(glm::vec3(-si, 0.f, co));
    const glm::vec3 nPort = -nStar;

    auto pushTri = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& n) {
        const glm::vec3 no = glm::normalize(n);
        for (const glm::vec3& p : {a, b, c}) {
            out.push_back(p.x);
            out.push_back(p.y);
            out.push_back(p.z);
            out.push_back(no.x);
            out.push_back(no.y);
            out.push_back(no.z);
        }
    };

    pushTri(v0, v2, v1, glm::vec3(0.f, -1.f, 0.f));
    pushTri(v0, v3, v2, glm::vec3(0.f, -1.f, 0.f));
    pushTri(v4, v5, v6, glm::vec3(0.f, 1.f, 0.f));
    pushTri(v4, v6, v7, glm::vec3(0.f, 1.f, 0.f));

    pushTri(v1, v2, v5, nBow);
    pushTri(v2, v6, v5, nBow);

    pushTri(v0, v4, v3, nStern);
    pushTri(v3, v4, v7, nStern);

    pushTri(v2, v3, v6, nStar);
    pushTri(v3, v7, v6, nStar);

    pushTri(v0, v1, v4, nPort);
    pushTri(v1, v5, v4, nPort);
}

}  // namespace

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Shallow water", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(kVsync ? 1 : 0);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "gladLoadGLLoader failed\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    glEnable(GL_FRAMEBUFFER_SRGB);

    FrameCtx frame;
    glfwSetWindowUserPointer(window, &frame);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);
    glfwGetFramebufferSize(window, &frame.fbW, &frame.fbH);

    std::string waterVs = loadTextFile(shaderPath("water_surface.vert"));
    std::string waterFs = loadTextFile(shaderPath("water_surface.frag"));
    if (waterVs.empty() || waterFs.empty()) {
        std::fprintf(stderr, "failed to load water shaders (tried %s)\n", shaderPath("water_surface.vert").c_str());
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    GLuint prog = makeProgram(waterVs.c_str(), waterFs.c_str());
    if (!prog) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    std::string terrainVs = loadTextFile(shaderPath("terrain.vert"));
    std::string terrainFs = loadTextFile(shaderPath("terrain.frag"));
    GLuint      terrainProg =
        (!terrainVs.empty() && !terrainFs.empty()) ? makeProgram(terrainVs.c_str(), terrainFs.c_str()) : 0u;
    if (kRenderTerrainMesh && !terrainProg)
        std::fprintf(stderr, "warning: terrain shaders missing (tried %s)\n", shaderPath("terrain.vert").c_str());
    GLint locTerrainMVP    = terrainProg ? glGetUniformLocation(terrainProg, "uMVP") : -1;
    GLint locTerrainLight  = terrainProg ? glGetUniformLocation(terrainProg, "uLightDir") : -1;
    GLint locTerrainCam    = terrainProg ? glGetUniformLocation(terrainProg, "uCamPos") : -1;
    GLint locTerrainDx     = terrainProg ? glGetUniformLocation(terrainProg, "uDx") : -1;
    GLint locTerrainHalfW  = terrainProg ? glGetUniformLocation(terrainProg, "uHalfW") : -1;
    GLint locTerrainHalfD  = terrainProg ? glGetUniformLocation(terrainProg, "uHalfD") : -1;
    GLint locTerrainTexB   = terrainProg ? glGetUniformLocation(terrainProg, "uB") : -1;
    GLint locTerrainUv     = terrainProg ? glGetUniformLocation(terrainProg, "uUvScale") : -1;
    GLint locTerrainAlbedo = terrainProg ? glGetUniformLocation(terrainProg, "uAlbedo") : -1;
    GLint locTerrainNrm    = terrainProg ? glGetUniformLocation(terrainProg, "uNormalMap") : -1;
    GLint locTerrainAO     = terrainProg ? glGetUniformLocation(terrainProg, "uAO") : -1;
    GLint locTerrainRough  = terrainProg ? glGetUniformLocation(terrainProg, "uRoughness") : -1;

    GLint locMVP = glGetUniformLocation(prog, "uMVP");
    GLint locLight = glGetUniformLocation(prog, "uLightDir");
    GLint locCamPos = glGetUniformLocation(prog, "uCameraPos");
    GLint locWaterAlpha = glGetUniformLocation(prog, "uAlpha");
    GLint locDx = glGetUniformLocation(prog, "uDx");
    GLint locHalfW = glGetUniformLocation(prog, "uHalfW");
    GLint locHalfD = glGetUniformLocation(prog, "uHalfD");
    GLint locTexH = glGetUniformLocation(prog, "uH");
    GLint locTexB = glGetUniformLocation(prog, "uB");
    GLint locEnvMap = glGetUniformLocation(prog, "uEnvMap");
    GLint locEnvMaxMip = glGetUniformLocation(prog, "uEnvMaxMip");
    GLint locWaveRough = glGetUniformLocation(prog, "uWaveRough");
    GLint locReflTex = glGetUniformLocation(prog, "uReflectionTex");
    GLint locReflVP  = glGetUniformLocation(prog, "uReflViewProj");
    GLint locWetEps  = glGetUniformLocation(prog, "uWetDepthEps");
    GLint locEtaRefU = glGetUniformLocation(prog, "uEtaRef");
    GLint locShoreBlendRange = glGetUniformLocation(prog, "uShoreBlendRange");
    GLint locWaterTime       = glGetUniformLocation(prog, "uTime");
    GLint locGerstnerWeight  = glGetUniformLocation(prog, "uGerstnerWeight");
    GLint locWaveScale       = glGetUniformLocation(prog, "uWaveScale");
    GLint locWaveStrength    = glGetUniformLocation(prog, "uWaveStrength");
    GLint locWaterAnimation  = glGetUniformLocation(prog, "uWaterAnimation");
    GLint locRefractTex      = glGetUniformLocation(prog, "uRefractionTex");
    GLint locViewport        = glGetUniformLocation(prog, "uViewport");
    GLint locViewRot         = glGetUniformLocation(prog, "uViewRot");
    GLint locWaterIOR        = glGetUniformLocation(prog, "uIOR");
    GLint locRefractParallax = glGetUniformLocation(prog, "uTransparency");
    GLint locWaterColor      = glGetUniformLocation(prog, "uWaterColor");
    GLint locWaterReflections = glGetUniformLocation(prog, "uWaterReflections");
    GLint locSceneDepth       = glGetUniformLocation(prog, "uSceneDepth");
    GLint locClipNF           = glGetUniformLocation(prog, "uClipNF");
    GLint locDepthAbsorb      = glGetUniformLocation(prog, "uDepthAbsorb");
    GLint locRefractionMaxOffset = glGetUniformLocation(prog, "uRefractionMaxOffset");
    GLint locRefractionLinTol    = glGetUniformLocation(prog, "uRefractionLinTol");

    Grid                      g(kNx, kNy, kDx, kDt);
    std::unique_ptr<Grid>     gCompareSwe;
    if (kSplitCompareSwe)
        gCompareSwe = std::make_unique<Grid>(kNx, kNy, kDx, kDt);

    const std::string terrainRawPath = std::string(BP_TOOLS_ROOT) + "/terrain_coastal.raw";
    if (loadTerrainHeightmapRaw(terrainRawPath.c_str(), g)) {
        std::printf("loaded terrain heightmap: %s\n", terrainRawPath.c_str());
        if (gCompareSwe)
            std::memcpy(gCompareSwe->terrain.data(), g.terrain.data(),
                        static_cast<size_t>(g.NX * g.NY) * sizeof(float));
    } else {
        std::fprintf(stderr, "terrain: file not found or wrong size (%s), using B=0\n",
                     terrainRawPath.c_str());
    }

    WaveDecomposition waveDec;
    std::vector<float> hTildePrevHalf;
    std::vector<float> hTildeSym;
    bool               haveHtildePrevHalf = false;
    std::unique_ptr<AiryEWaveFFTW> airy;
    if (kCoupledStep || kSplitCompareSwe)
        airy = std::make_unique<AiryEWaveFFTW>(kNx, kNy, kDx);
    const float halfW = 0.5f * kNx * kDx;
    const float halfD = 0.5f * kNy * kDx;

    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            const float hInit = fmaxf(0.f, kEtaRef - g.B(i, j));
            g.H(i, j) = hInit;
            if (gCompareSwe)
                gCompareSwe->H(i, j) = hInit;
        }
    }

    std::vector<unsigned> indices;
    buildGridIndices(kNx, kNy, indices);
    std::vector<float> waterCornerIJ;
    buildWaterCornerIJ(kNx, kNy, waterCornerIJ);

    GLuint texH = 0, texB = 0;
    allocGridTextures(kNx, kNy, texH, texB);
    uploadTerrainTexture(g, texB);

    TerrainSand04Textures sandTex{};
    loadTerrainSand04(BP_SAND01_ROOT, sandTex);

    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(waterCornerIJ.size() * sizeof(float)), waterCornerIJ.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned)),
                 indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    std::string boatVs = loadTextFile(shaderPath("boat.vert"));
    std::string boatFs = loadTextFile(shaderPath("boat.frag"));
    GLuint      boatProg = 0;
    if (!boatVs.empty() && !boatFs.empty())
        boatProg = makeProgram(boatVs.c_str(), boatFs.c_str());
    if (!boatProg)
        std::fprintf(stderr, "warning: boat shaders missing or failed to compile (tried %s)\n",
                     shaderPath("boat.vert").c_str());
    GLint locBoatMVP = boatProg ? glGetUniformLocation(boatProg, "uMVP") : -1;
    GLint locBoatLight = boatProg ? glGetUniformLocation(boatProg, "uLightDir") : -1;
    GLint locBoatColor = boatProg ? glGetUniformLocation(boatProg, "uBaseColor") : -1;
    GLint locBoatClipRefl = boatProg ? glGetUniformLocation(boatProg, "uClipRefl") : -1;
    GLint locBoatWaterY = boatProg ? glGetUniformLocation(boatProg, "uWaterPlaneY") : -1;
    GLuint boatVao = 0, boatVbo = 0;
    glGenVertexArrays(1, &boatVao);
    glGenBuffers(1, &boatVbo);
    glBindVertexArray(boatVao);
    glBindBuffer(GL_ARRAY_BUFFER, boatVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 256, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    std::string skyVs = loadTextFile(shaderPath("skybox.vert"));
    std::string skyFs = loadTextFile(shaderPath("skybox.frag"));
    GLuint      skyProg = 0;
    if (!skyVs.empty() && !skyFs.empty())
        skyProg = makeProgram(skyVs.c_str(), skyFs.c_str());
    if (!skyProg)
        std::fprintf(stderr, "warning: skybox shaders missing or failed to compile (tried %s)\n",
                     shaderPath("skybox.vert").c_str());
    GLint locSkyProj = skyProg ? glGetUniformLocation(skyProg, "uProj") : -1;
    GLint locSkyView = skyProg ? glGetUniformLocation(skyProg, "uViewSky") : -1;
    GLint locSkyMap  = skyProg ? glGetUniformLocation(skyProg, "uSkyMap") : -1;
    GLuint skyVao = 0, skyVbo = 0;
    glGenVertexArrays(1, &skyVao);
    glGenBuffers(1, &skyVbo);
    glBindVertexArray(skyVao);
    glBindBuffer(GL_ARRAY_BUFFER, skyVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kSkyboxVerts), kSkyboxVerts, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CLIP_DISTANCE0);
    glClearColor(0.14f, 0.18f, 0.26f, 1.f);

    GLuint reflFbo = 0, reflColorTex = 0, reflDepthRbo = 0;
    int    reflBufW = 0, reflBufH = 0;
    GLuint refractColorTex = 0;
    int    refractBufW = 0, refractBufH = 0;
    GLuint refractDepthFbo = 0, refractDepthTex = 0;
    int    refractDepthBufW = 0, refractDepthBufH = 0;

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.35f, 0.85f, 0.4f));

    float     envMaxMipF = 0.f;
    const GLuint envCubemap = createEnvCubemap(BP_SKYBOX_ROOT, lightDir, &envMaxMipF);

    Boat boat;
    {
        const float hL  = boat.length * 0.5f;
        const float hW  = boat.width * 0.5f;
        const float ext = std::sqrt(hL * hL + hW * hW);
        const float pad = kDx;
        boat.pos = glm::vec2(-halfW + ext + pad, 0.f);
    }
    boat.heading = 0.f;
    boat.throttle = 0.f;
    std::vector<float> boatVerts;
    boatVerts.reserve(256);

    double fpsPrevT = glfwGetTime();
    float  fpsShown = 0.f;
    double simT     = 0.0;

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        const bool manualControl = true;
        for (int s = 0; s < kSubsteps; ++s) {
            updateBoat(boat, g, window, halfW, halfD, g.dt, manualControl);
            applyBoatForcing(boat, g, halfW, halfD, g.dt);
            if (kSplitCompareSwe) {
                applyBoatForcing(boat, *gCompareSwe, halfW, halfD, gCompareSwe->dt);
                coupledSubstep(g, halfW, halfD, waveDec, *airy, hTildeSym, hTildePrevHalf, haveHtildePrevHalf,
                                 kGradPenaltyD, 0.25f, kWaveDiffuseIters);
                sweStepGpu(*gCompareSwe);
            } else if (kCoupledStep) {
                coupledSubstep(g, halfW, halfD, waveDec, *airy, hTildeSym, hTildePrevHalf, haveHtildePrevHalf,
                                 kGradPenaltyD, 0.25f, kWaveDiffuseIters);
            } else {
                sweStepGpu(g);
            }
        }
        simT += kSubsteps * g.dt;

        static double lastFroudePrintSimT = -1e9;
        constexpr double kFroudePrintInterval = 0.25;
        if (simT - lastFroudePrintSimT >= kFroudePrintInterval) {
            lastFroudePrintSimT = simT;
            if (kSplitCompareSwe && gCompareSwe) {
                const ShallowWaterDiagnostics dCoupled = gridShallowWaterDiagnostics(g, 9.81f);
                const ShallowWaterDiagnostics dSwe    = gridShallowWaterDiagnostics(*gCompareSwe, 9.81f);
                std::printf(
                    "t=%.3f s  coupled Fr_max=%.4f  |u|_max=%.4f  h_min_wet=%.4f m  |  SWE Fr_max=%.4f  "
                    "|u|_max=%.4f  h_min_wet=%.4f m\n",
                    simT, static_cast<double>(dCoupled.fr_max), static_cast<double>(dCoupled.speed_max),
                    static_cast<double>(dCoupled.h_min_wet), static_cast<double>(dSwe.fr_max),
                    static_cast<double>(dSwe.speed_max), static_cast<double>(dSwe.h_min_wet));
            } else {
                const ShallowWaterDiagnostics diag = gridShallowWaterDiagnostics(g, 9.81f);
                std::printf(
                    "t=%.3f s  Fr_max=%.4f  |u|@Frmax=%.4f m/s  h@Frmax=%.4f m  |u|_max=%.4f  h_min_wet=%.4f m\n",
                    simT, static_cast<double>(diag.fr_max), static_cast<double>(diag.speed_at_fr_max),
                    static_cast<double>(diag.h_at_fr_max), static_cast<double>(diag.speed_max),
                    static_cast<double>(diag.h_min_wet));
            }
            std::fflush(stdout);
        }

        glm::mat4 view =
            glm::lookAt(kFixedCamEye, kFixedCamTarget, glm::vec3(0.f, 1.f, 0.f));

        auto drawPane = [&](int vpX, int vpW, const Grid& grid, float aspect) {
            glm::mat4 proj      = glm::perspective(glm::radians(kCamFovDeg), aspect, 0.1f, 500.f);
            glm::mat4 viewRefl  = reflectionViewAcrossY(kReflectPlaneY, kFixedCamEye, kFixedCamTarget);
            glm::mat4 reflVP    = proj * viewRefl;
            glm::mat4 mvp       = proj * view;

            uploadWaterDepthTexture(grid, texH);
            fillBoatSolidMesh(boatVerts, boat);

            ensureReflectionFBO(vpW, frame.fbH, reflFbo, reflColorTex, reflDepthRbo, reflBufW, reflBufH);
            if (reflFbo) {
                glBindFramebuffer(GL_FRAMEBUFFER, reflFbo);
                glViewport(0, 0, vpW, frame.fbH);
                glClearColor(0.04f, 0.06f, 0.1f, 1.f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                glEnable(GL_DEPTH_TEST);

                if (skyProg) {
                    glDepthMask(GL_FALSE);
                    glDepthFunc(GL_LEQUAL);
                    glUseProgram(skyProg);
                    const glm::mat4 skyViewR = glm::mat4(glm::mat3(viewRefl));
                    if (locSkyProj >= 0)
                        glUniformMatrix4fv(locSkyProj, 1, GL_FALSE, glm::value_ptr(proj));
                    if (locSkyView >= 0)
                        glUniformMatrix4fv(locSkyView, 1, GL_FALSE, glm::value_ptr(skyViewR));
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
                    if (locSkyMap >= 0)
                        glUniform1i(locSkyMap, 0);
                    glBindVertexArray(skyVao);
                    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(sizeof(kSkyboxVerts) / (sizeof(float) * 3u)));
                    glBindVertexArray(0);
                    glDepthFunc(GL_LESS);
                    glDepthMask(GL_TRUE);
                }

                if (boatProg) {
                    glUseProgram(boatProg);
                    const glm::mat4 mvpR = proj * viewRefl;
                    glUniformMatrix4fv(locBoatMVP, 1, GL_FALSE, glm::value_ptr(mvpR));
                    glUniform3fv(locBoatLight, 1, glm::value_ptr(lightDir));
                    const glm::vec3 hullColor(0.78f, 0.44f, 0.2f);
                    glUniform3fv(locBoatColor, 1, glm::value_ptr(hullColor));
                    if (locBoatClipRefl >= 0)
                        glUniform1i(locBoatClipRefl, 1);
                    if (locBoatWaterY >= 0)
                        glUniform1f(locBoatWaterY, kReflectPlaneY);
                    glBindVertexArray(boatVao);
                    glBindBuffer(GL_ARRAY_BUFFER, boatVbo);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(boatVerts.size() * sizeof(float)),
                                    boatVerts.data());
                    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(boatVerts.size() / 6));
                    glBindVertexArray(0);
                    if (locBoatClipRefl >= 0)
                        glUniform1i(locBoatClipRefl, 0);
                }

                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }

            glViewport(vpX, 0, vpW, frame.fbH);

            if (skyProg) {
                glDepthMask(GL_FALSE);
                glDepthFunc(GL_LEQUAL);
                glUseProgram(skyProg);
                const glm::mat4 skyView = glm::mat4(glm::mat3(view));
                if (locSkyProj >= 0)
                    glUniformMatrix4fv(locSkyProj, 1, GL_FALSE, glm::value_ptr(proj));
                if (locSkyView >= 0)
                    glUniformMatrix4fv(locSkyView, 1, GL_FALSE, glm::value_ptr(skyView));
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
                if (locSkyMap >= 0)
                    glUniform1i(locSkyMap, 0);
                glBindVertexArray(skyVao);
                glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(sizeof(kSkyboxVerts) / (sizeof(float) * 3u)));
                glBindVertexArray(0);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
            }

            if (kRenderTerrainMesh && terrainProg) {
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
                glUseProgram(terrainProg);
                if (locTerrainMVP >= 0)
                    glUniformMatrix4fv(locTerrainMVP, 1, GL_FALSE, glm::value_ptr(mvp));
                if (locTerrainLight >= 0)
                    glUniform3fv(locTerrainLight, 1, glm::value_ptr(lightDir));
                if (locTerrainCam >= 0)
                    glUniform3fv(locTerrainCam, 1, glm::value_ptr(kFixedCamEye));
                if (locTerrainDx >= 0)
                    glUniform1f(locTerrainDx, kDx);
                if (locTerrainHalfW >= 0)
                    glUniform1f(locTerrainHalfW, halfW);
                if (locTerrainHalfD >= 0)
                    glUniform1f(locTerrainHalfD, halfD);
                if (locTerrainUv >= 0)
                    glUniform1f(locTerrainUv, kTerrainMaterialUvScale);
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texB);
                if (locTerrainTexB >= 0)
                    glUniform1i(locTerrainTexB, 0);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, sandTex.albedo);
                if (locTerrainAlbedo >= 0)
                    glUniform1i(locTerrainAlbedo, 1);
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, sandTex.normalGl);
                if (locTerrainNrm >= 0)
                    glUniform1i(locTerrainNrm, 2);
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_2D, sandTex.ao);
                if (locTerrainAO >= 0)
                    glUniform1i(locTerrainAO, 3);
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_2D, sandTex.roughness);
                if (locTerrainRough >= 0)
                    glUniform1i(locTerrainRough, 4);
                glActiveTexture(GL_TEXTURE0);
                glBindVertexArray(vao);
                glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
                glBindVertexArray(0);
            }

            ensureRefractTex(vpW, frame.fbH, refractColorTex, refractBufW, refractBufH);
            if (refractColorTex) {
                glBindTexture(GL_TEXTURE_2D, refractColorTex);
                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vpX, 0, vpW, frame.fbH);
                glBindTexture(GL_TEXTURE_2D, 0);
            }
            ensureRefractDepthFbo(vpW, frame.fbH, refractDepthFbo, refractDepthTex, refractDepthBufW,
                                  refractDepthBufH);
            if (refractDepthFbo) {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, refractDepthFbo);
                glBlitFramebuffer(vpX, 0, vpX + vpW, frame.fbH, 0, 0, vpW, frame.fbH, GL_DEPTH_BUFFER_BIT,
                                  GL_NEAREST);
                glBindFramebuffer(GL_FRAMEBUFFER, 0);
            }

            if (boatProg) {
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
                glUseProgram(boatProg);
                glUniformMatrix4fv(locBoatMVP, 1, GL_FALSE, glm::value_ptr(mvp));
                glUniform3fv(locBoatLight, 1, glm::value_ptr(lightDir));
                const glm::vec3 hullColor(0.78f, 0.44f, 0.2f);
                glUniform3fv(locBoatColor, 1, glm::value_ptr(hullColor));
                if (locBoatClipRefl >= 0)
                    glUniform1i(locBoatClipRefl, 0);
                glBindVertexArray(boatVao);
                glBindBuffer(GL_ARRAY_BUFFER, boatVbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(boatVerts.size() * sizeof(float)),
                                boatVerts.data());
                glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(boatVerts.size() / 6));
                glBindVertexArray(0);
            }

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glDepthMask(GL_FALSE);
            glDisable(GL_CULL_FACE);

            glUseProgram(prog);
            glUniformMatrix4fv(locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform3fv(locLight, 1, glm::value_ptr(lightDir));
            if (locCamPos >= 0)
                glUniform3fv(locCamPos, 1, glm::value_ptr(kFixedCamEye));
            if (locReflVP >= 0)
                glUniformMatrix4fv(locReflVP, 1, GL_FALSE, glm::value_ptr(reflVP));
            if (locWaterAlpha >= 0)
                glUniform1f(locWaterAlpha, kWaterAlpha);
            if (locDx >= 0)
                glUniform1f(locDx, kDx);
            if (locHalfW >= 0)
                glUniform1f(locHalfW, halfW);
            if (locHalfD >= 0)
                glUniform1f(locHalfD, halfD);
            if (locWetEps >= 0)
                glUniform1f(locWetEps, kWetDepthEps);
            if (locEtaRefU >= 0)
                glUniform1f(locEtaRefU, kEtaRef);
            if (locShoreBlendRange >= 0)
                glUniform1f(locShoreBlendRange, kShoreBlendRange);
            if (locWaterTime >= 0)
                glUniform1f(locWaterTime, static_cast<float>(glfwGetTime()));
            if (locGerstnerWeight >= 0)
                glUniform1f(locGerstnerWeight, kGerstnerWeight);
            if (locWaveScale >= 0)
                glUniform1f(locWaveScale, kWaterWaveScale);
            if (locWaveStrength >= 0)
                glUniform1f(locWaveStrength, kWaterWaveStrength);
            if (locWaterAnimation >= 0)
                glUniform1f(locWaterAnimation, kWaterAnimationSpeed);
            if (locViewport >= 0)
                glUniform4f(locViewport, static_cast<float>(vpX), 0.f, static_cast<float>(vpW),
                            static_cast<float>(frame.fbH));
            if (locViewRot >= 0) {
                const glm::mat3 viewRot(view);
                glUniformMatrix3fv(locViewRot, 1, GL_FALSE, glm::value_ptr(viewRot));
            }
            if (locWaterIOR >= 0)
                glUniform1f(locWaterIOR, kWaterIOR);
            if (locRefractParallax >= 0)
                glUniform1f(locRefractParallax, kWaterRefractionParallax);
            if (locWaterColor >= 0)
                glUniform3fv(locWaterColor, 1, glm::value_ptr(kWaterColor));
            if (locWaterReflections >= 0)
                glUniform1f(locWaterReflections, kWaterReflections);
            if (locClipNF >= 0)
                glUniform2f(locClipNF, kClipNear, kClipFar);
            if (locDepthAbsorb >= 0)
                glUniform1f(locDepthAbsorb, kDepthAbsorb);
            if (locRefractionMaxOffset >= 0)
                glUniform1f(locRefractionMaxOffset, kRefractionMaxOffset);
            if (locRefractionLinTol >= 0)
                glUniform1f(locRefractionLinTol, kRefractionLinTol);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texH);
            if (locTexH >= 0)
                glUniform1i(locTexH, 0);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, texB);
            if (locTexB >= 0)
                glUniform1i(locTexB, 1);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
            if (locEnvMap >= 0)
                glUniform1i(locEnvMap, 2);
            if (locEnvMaxMip >= 0)
                glUniform1f(locEnvMaxMip, envMaxMipF);
            if (locWaveRough >= 0)
                glUniform1f(locWaveRough, kEnvWaveRough);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, reflColorTex);
            if (locReflTex >= 0)
                glUniform1i(locReflTex, 3);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, refractColorTex);
            if (locRefractTex >= 0)
                glUniform1i(locRefractTex, 4);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_2D, refractDepthTex);
            if (locSceneDepth >= 0)
                glUniform1i(locSceneDepth, 5);
            glActiveTexture(GL_TEXTURE0);

            glBindVertexArray(vao);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
            glBindVertexArray(0);

            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
        };

        glViewport(0, 0, frame.fbW, frame.fbH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDepthMask(GL_TRUE);
        glDisable(GL_CULL_FACE);

        if (kSplitCompareSwe && gCompareSwe) {
            const int   halfFbW = frame.fbW / 2;
            const int   restW   = frame.fbW - halfFbW;
            const float aspectL = frame.fbH > 0 ? static_cast<float>(halfFbW) / static_cast<float>(frame.fbH) : 1.f;
            const float aspectR = frame.fbH > 0 ? static_cast<float>(restW) / static_cast<float>(frame.fbH) : 1.f;
            drawPane(0, halfFbW, g, aspectL);
            drawPane(halfFbW, restW, *gCompareSwe, aspectR);
        } else {
            const float aspect =
                frame.fbH > 0 ? static_cast<float>(frame.fbW) / static_cast<float>(frame.fbH) : 1.f;
            drawPane(0, frame.fbW, g, aspect);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        {
            const double now = glfwGetTime();
            const double dtF = now - fpsPrevT;
            fpsPrevT = now;
            if (dtF > 1e-6 && dtF < 2.0) {
                const float inst = static_cast<float>(1.0 / dtF);
                fpsShown = (fpsShown < 1e-3f) ? inst : (fpsShown * 0.92f + inst * 0.08f);
            }
            char title[256];
            const char* gear = (boat.speed > 0.05f) ? "FWD" : (boat.speed < -0.05f) ? "REV" : "NEU";
            if (kSplitCompareSwe) {
                std::snprintf(title, sizeof title,
                              "Coupled left | SWE right | %.0f FPS | %.2f m/s (%s) | thr %.2f",
                              static_cast<double>(fpsShown), boat.speed, gear, boat.throttle);
            } else {
                std::snprintf(title, sizeof title,
                              "Shallow water | %.0f FPS | %.2f m/s (%s) | throttle %.2f",
                              static_cast<double>(fpsShown), boat.speed, gear, boat.throttle);
            }
            glfwSetWindowTitle(window, title);
        }
    }

    glDeleteProgram(prog);
    if (terrainProg)
        glDeleteProgram(terrainProg);
    if (boatProg)
        glDeleteProgram(boatProg);
    if (skyProg)
        glDeleteProgram(skyProg);
    glDeleteBuffers(1, &skyVbo);
    glDeleteVertexArrays(1, &skyVao);
    glDeleteTextures(1, &texH);
    glDeleteTextures(1, &texB);
    destroyTerrainSand04(sandTex);
    glDeleteTextures(1, &envCubemap);
    if (reflFbo) {
        glDeleteFramebuffers(1, &reflFbo);
        glDeleteTextures(1, &reflColorTex);
        glDeleteRenderbuffers(1, &reflDepthRbo);
    }
    if (refractColorTex)
        glDeleteTextures(1, &refractColorTex);
    if (refractDepthFbo) {
        glDeleteFramebuffers(1, &refractDepthFbo);
        glDeleteTextures(1, &refractDepthTex);
    }
    glDeleteBuffers(1, &boatVbo);
    glDeleteVertexArrays(1, &boatVao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
