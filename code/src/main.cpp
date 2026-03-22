#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "airy_fftw.h"
#include "boat.h"
#include "jw_pipeline.h"
#include "shallow_water_solver.h"
#include "wavedecomposer.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

constexpr int   kNx = 128;
constexpr int   kNy = 128;
constexpr float kDx = 1.0f;
constexpr float kDt = 1.0f / 120.0f;
constexpr int   kSubsteps = 2;
// J&W 2023 Algorithm 1：步初分解 → bulk SWE → Airy → 表面输运 → 合成（见 jw_pipeline.cpp）
constexpr bool  kUseAiryEWave = true;
constexpr float kGradPenaltyD = 0.01f; // 论文式 (13) 中 d = 1/100

static const char* kVertSrc = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float aDepth;
uniform mat4 uMVP;
out vec3 vWorldPos;
out float vDepth;
void main() {
    vWorldPos = aPos;
    vDepth = aDepth;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char* kFragSrc = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in float vDepth;
uniform vec3 uLightDir;
uniform float uAlpha;
out vec4 FragColor;
void main() {
    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    float t = clamp(vDepth / 4.0, 0.0, 1.0);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.14, 0.32);
    vec3 base = mix(deep, shallow, t);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = base * (0.22 + 0.78 * ndl);
    FragColor = vec4(rgb, uAlpha);
}
)GLSL";

static const char* kBoatVert = R"GLSL(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
uniform mat4 uMVP;
out vec3 vNrm;
void main() {
    vNrm = aNrm;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char* kBoatFrag = R"GLSL(
#version 330 core
in vec3 vNrm;
uniform vec3 uLightDir;
uniform vec3 uBaseColor;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNrm);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = uBaseColor * (0.28 + 0.72 * ndl);
    FragColor = vec4(rgb, 1.0);
}
)GLSL";

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

float cornerSurfaceY(const Grid& g, int vi, int vj) {
    float s = 0.f;
    int   n = 0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ci = vi + di;
            int cj = vj + dj;
            if (ci >= 0 && ci < g.NX && cj >= 0 && cj < g.NY) {
                s += g.B(ci, cj) + g.H(ci, cj);
                ++n;
            }
        }
    }
    return n ? s / static_cast<float>(n) : 0.f;
}

float cornerAvgH(const Grid& g, int vi, int vj) {
    float s = 0.f;
    int   n = 0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ci = vi + di;
            int cj = vj + dj;
            if (ci >= 0 && ci < g.NX && cj >= 0 && cj < g.NY) {
                s += g.H(ci, cj);
                ++n;
            }
        }
    }
    return n ? s / static_cast<float>(n) : 0.f;
}

void fillWaterMesh(const Grid& g, float halfW, float halfD, std::vector<float>& interleaved) {
    const int vx = g.NX + 1;
    const int vz = g.NY + 1;
    interleaved.resize(static_cast<size_t>(vx * vz * 4));
    size_t k = 0;
    for (int j = 0; j < vz; ++j) {
        for (int i = 0; i < vx; ++i) {
            float x = i * g.dx - halfW;
            float z = j * g.dx - halfD;
            float y = cornerSurfaceY(g, i, j);
            float h = cornerAvgH(g, i, j);
            interleaved[k++] = x;
            interleaved[k++] = y;
            interleaved[k++] = z;
            interleaved[k++] = h;
        }
    }
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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Shallow water", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "glfwCreateWindow failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::fprintf(stderr, "gladLoadGLLoader failed\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    FrameCtx frame;
    glfwSetWindowUserPointer(window, &frame);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);
    glfwGetFramebufferSize(window, &frame.fbW, &frame.fbH);

    GLuint prog = makeProgram(kVertSrc, kFragSrc);
    if (!prog) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    GLint locMVP = glGetUniformLocation(prog, "uMVP");
    GLint locLight = glGetUniformLocation(prog, "uLightDir");
    GLint locWaterAlpha = glGetUniformLocation(prog, "uAlpha");

    Grid g(kNx, kNy, kDx, kDt);
    WaveDecomposition waveDec;
    // Airy：\tilde h 与 q 在整步 t 对齐 — (h^{t-Δt/2}+h^{t+Δt/2})/2，来自相邻两次波分解
    std::vector<float> hTildePrevHalf;
    std::vector<float> hTildeSym;
    bool               haveHtildePrevHalf = false;
    std::unique_ptr<AiryEWaveFFTW> airy;
    if (kUseAiryEWave)
        airy = std::make_unique<AiryEWaveFFTW>(kNx, kNy, kDx);
    const float halfW = 0.5f * kNx * kDx;
    const float halfD = 0.5f * kNy * kDx;

    const float h0 = 4.0f;
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i)
            g.H(i, j) = h0;
    }

    std::vector<unsigned> indices;
    buildGridIndices(kNx, kNy, indices);
    std::vector<float> vertices;

    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned)),
                 indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    GLuint boatProg = makeProgram(kBoatVert, kBoatFrag);
    GLint locBoatMVP = boatProg ? glGetUniformLocation(boatProg, "uMVP") : -1;
    GLint locBoatLight = boatProg ? glGetUniformLocation(boatProg, "uLightDir") : -1;
    GLint locBoatColor = boatProg ? glGetUniformLocation(boatProg, "uBaseColor") : -1;
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

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.14f, 0.18f, 0.26f, 1.f);

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.35f, 0.85f, 0.4f));

    Boat boat;
    boat.pos = glm::vec2(-18.f, 0.f);
    boat.heading = 0.f;
    boat.throttle = 0.55f;
    std::vector<float> boatVerts;
    boatVerts.reserve(256);

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        const bool manualControl = true;
        for (int s = 0; s < kSubsteps; ++s) {
            updateBoat(boat, g, window, halfW, halfD, g.dt, manualControl);
            applyBoatForcing(boat, g, halfW, halfD, g.dt);
            if (airy)
                jwCoupledSubstep(g, halfW, halfD, waveDec, *airy, hTildeSym, hTildePrevHalf, haveHtildePrevHalf,
                                 kGradPenaltyD);
            else
                sweStep(g);
        }

        {
            char title[192];
            const char* gear = (boat.speed > 0.05f) ? "FWD" : (boat.speed < -0.05f) ? "REV" : "NEU";
            std::snprintf(title, sizeof title, "Shallow water | %.2f m/s (%s) | throttle %.2f", boat.speed, gear,
                          boat.throttle);
            glfwSetWindowTitle(window, title);
        }

        fillWaterMesh(g, halfW, halfD, vertices);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
                     vertices.data(), GL_DYNAMIC_DRAW);

        float aspect = frame.fbH > 0 ? static_cast<float>(frame.fbW) / static_cast<float>(frame.fbH) : 1.f;
        const float ang = 0.85f;  // fixed azimuth (no orbit)
        const float rad = 75.f;
        const float eyeY = 38.f;
        glm::vec3 eye(rad * std::cos(ang), eyeY, rad * std::sin(ang));
        glm::mat4 proj = glm::perspective(glm::radians(50.f), aspect, 0.1f, 500.f);
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0.f, 3.5f, 0.f), glm::vec3(0.f, 1.f, 0.f));
        glm::mat4 mvp = proj * view;

        glViewport(0, 0, frame.fbW, frame.fbH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDepthMask(GL_TRUE);
        glDisable(GL_CULL_FACE);
        if (boatProg) {
            fillBoatSolidMesh(boatVerts, boat);

            glUseProgram(boatProg);
            glUniformMatrix4fv(locBoatMVP, 1, GL_FALSE, glm::value_ptr(mvp));
            glUniform3fv(locBoatLight, 1, glm::value_ptr(lightDir));
            const glm::vec3 hullColor(0.78f, 0.44f, 0.2f);
            glUniform3fv(locBoatColor, 1, glm::value_ptr(hullColor));
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
        if (locWaterAlpha >= 0)
            glUniform1f(locWaterAlpha, 0.84f);
        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteProgram(prog);
    if (boatProg)
        glDeleteProgram(boatProg);
    glDeleteBuffers(1, &boatVbo);
    glDeleteVertexArrays(1, &boatVao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
