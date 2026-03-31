#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_file.h"
#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kNx = 256;
constexpr int kNy = 64;
constexpr float kDx = 1.0f;
// Slightly conservative vs dx=1 and face CFL cap (see shallow_water_solver_gpu.cu faceSpeedCap_d).
constexpr float kDt = 1.0f / 120.0f;
constexpr int kSubsteps = 2;
constexpr bool kVsync = true;
constexpr int kWallThicknessCells = 2;
constexpr int kGapHalfWidthCells = 6;
// Negative moves wall toward upstream (left), positive toward downstream (right).
constexpr int kWallCenterOffsetCells = -30;
constexpr int kSecondWallCenterOffsetCells = -4;
constexpr int kThirdWallCenterOffsetCells = 16;
constexpr float kWallHeight = 10.f;
// Vertex: cell counts as wet if h > this. Lower = thin front visible (matches fragment fade).
constexpr float kWetDepthEps = 1e-2f;
constexpr bool kUseCenterGap = true;
constexpr float kG = 9.81f;
constexpr float kGapRoundnessXCells = 1.2f;
constexpr int kGapCount = 3;        // first wall
constexpr int kSecondGapCount = 1;  // second wall: one centered opening
constexpr int kThirdGapCount = 0;   // third wall: fully closed
constexpr int kGapSpacingCells = 18;
// Second wall breach larger than the first (ellipse half-axes in cell units).
constexpr float kSecondGapRoundnessXCells = 2.0f;
constexpr float kSecondGapHalfWidthCells = 11.f;
constexpr glm::vec3 kFixedCamPos(44.03f, 61.59f, -2.16f);
constexpr float kFixedCamYawDeg = -180.48f;
constexpr float kFixedCamPitchDeg = -49.76f;
constexpr float kFixedCamFovDeg = 32.80f;

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
            idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
            idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
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

void uploadGridTextures(const Grid& g, GLuint texH, GLuint texB) {
    glBindTexture(GL_TEXTURE_2D, texH);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.NX, g.NY, GL_RED, GL_FLOAT, g.h.data());
    glBindTexture(GL_TEXTURE_2D, texB);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, g.NX, g.NY, GL_RED, GL_FLOAT, g.terrain.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void pushQuad(std::vector<float>& out, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d,
              const glm::vec3& n) {
    const glm::vec3 no = glm::normalize(n);
    auto pushV = [&](const glm::vec3& p) {
        out.push_back(p.x); out.push_back(p.y); out.push_back(p.z);
        out.push_back(no.x); out.push_back(no.y); out.push_back(no.z);
    };
    pushV(a); pushV(b); pushV(c);
    pushV(a); pushV(c); pushV(d);
}

void addBox(std::vector<float>& out, float x0, float x1, float y0, float y1, float z0, float z1) {
    const glm::vec3 p000(x0, y0, z0);
    const glm::vec3 p001(x0, y0, z1);
    const glm::vec3 p010(x0, y1, z0);
    const glm::vec3 p011(x0, y1, z1);
    const glm::vec3 p100(x1, y0, z0);
    const glm::vec3 p101(x1, y0, z1);
    const glm::vec3 p110(x1, y1, z0);
    const glm::vec3 p111(x1, y1, z1);

    pushQuad(out, p010, p110, p111, p011, glm::vec3(0.f, 1.f, 0.f));   // top
    pushQuad(out, p000, p001, p101, p100, glm::vec3(0.f, -1.f, 0.f));  // bottom
    pushQuad(out, p000, p100, p110, p010, glm::vec3(0.f, 0.f, -1.f));  // -z
    pushQuad(out, p001, p011, p111, p101, glm::vec3(0.f, 0.f, 1.f));   // +z
    pushQuad(out, p000, p010, p011, p001, glm::vec3(-1.f, 0.f, 0.f));  // -x
    pushQuad(out, p100, p101, p111, p110, glm::vec3(1.f, 0.f, 0.f));   // +x
}

bool isWallSolidCell(int i, int j, int nx, int ny) {
    const int wallCenters[3] = {
        nx / 2 + kWallCenterOffsetCells,
        nx / 2 + kSecondWallCenterOffsetCells,
        nx / 2 + kThirdWallCenterOffsetCells
    };
    for (int wc = 0; wc < 3; ++wc) {
        const int wallCenter = wallCenters[wc];
        const int wallI0 = wallCenter - kWallThicknessCells / 2;
        const int wallI1 = wallI0 + kWallThicknessCells;
        if (i < wallI0 || i >= wallI1) {
            continue;
        }
        if (!kUseCenterGap) {
            return true;
        }

        // Rounded openings: multiple ellipses in (x,z) over cell centers.
        const float cx = 0.5f * static_cast<float>(wallI0 + wallI1 - 1);
        const float rx = std::max(0.75f, wc == 1 ? kSecondGapRoundnessXCells : kGapRoundnessXCells);
        const float rz = std::max(1.0f, wc == 1 ? kSecondGapHalfWidthCells : static_cast<float>(kGapHalfWidthCells));
        const float centerZ = 0.5f * static_cast<float>(ny - 1);
        const int wallGapCount = (wc == 0) ? kGapCount : (wc == 1 ? kSecondGapCount : kThirdGapCount);
        if (wallGapCount <= 0) {
            return true;
        }
        const int gapCount = std::max(1, wallGapCount);
        const float spacing = static_cast<float>(std::max(1, kGapSpacingCells));

        bool inRoundedGap = false;
        for (int gi = 0; gi < gapCount; ++gi) {
            const float offsetIdx = static_cast<float>(gi) - 0.5f * static_cast<float>(gapCount - 1);
            const float cz = centerZ + offsetIdx * spacing;
            const float dx = (static_cast<float>(i) - cx) / rx;
            const float dz = (static_cast<float>(j) - cz) / rz;
            if (dx * dx + dz * dz <= 1.0f) {
                inRoundedGap = true;
                break;
            }
        }
        return !inRoundedGap;
    }
    return false;
}

void buildDamWallMesh(std::vector<float>& out, int nx, int ny, float dx, float halfW, float halfD) {
    out.clear();
    out.reserve(static_cast<size_t>(kWallThicknessCells * ny * 36u * 6u));

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (!isWallSolidCell(i, j, nx, ny)) {
                continue;
            }
            const float x0 = static_cast<float>(i) * dx - halfW;
            const float x1 = static_cast<float>(i + 1) * dx - halfW;
            const float z0 = static_cast<float>(j) * dx - halfD;
            const float z1 = static_cast<float>(j + 1) * dx - halfD;
            addBox(out, x0, x1, 0.0f, kWallHeight, z0, z1);
        }
    }
}

void setupDamInitialState(Grid& g) {
    const int wallCenter = g.NX / 2 + kWallCenterOffsetCells;
    const int wallI0 = wallCenter - kWallThicknessCells / 2;
    const float hLeft = 4.f;
    const float hRight = 5e-3f;
    const float transitionCells = 3.0f;
    const float x0 = static_cast<float>(wallI0);

    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            g.B(i, j) = 0.0f;
            const float x = static_cast<float>(i) - x0;
            const float blend = 0.5f * (1.0f + std::tanh(x / transitionCells));
            g.H(i, j) = hLeft * (1.0f - blend) + hRight * blend;

            if (isWallSolidCell(i, j, g.NX, g.NY)) {
                g.B(i, j) = kWallHeight;
                g.H(i, j) = 0.0f;
            }
        }
    }
    std::fill(g.qx.begin(), g.qx.end(), 0.0f);
    std::fill(g.qy.begin(), g.qy.end(), 0.0f);
    sweApplyBoundaryConditionsGpu(g);
}

struct FrameCtx {
    int fbW = 1280;
    int fbH = 720;
};

void framebufferSizeCB(GLFWwindow* w, int width, int height) {
    auto* ctx = static_cast<FrameCtx*>(glfwGetWindowUserPointer(w));
    if (ctx) {
        ctx->fbW = width;
        ctx->fbH = height;
    }
    glViewport(0, 0, width, height);
}

} // namespace

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Pure SWE Dam Break", nullptr, nullptr);
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

    FrameCtx frame;
    glfwSetWindowUserPointer(window, &frame);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCB);
    glfwGetFramebufferSize(window, &frame.fbW, &frame.fbH);

    std::string damWvs = loadTextFile(shaderPath("dam_water.vert"));
    std::string damWfs = loadTextFile(shaderPath("dam_water.frag"));
    if (damWvs.empty() || damWfs.empty()) {
        std::fprintf(stderr, "failed to load dam water shaders (tried %s)\n", shaderPath("dam_water.vert").c_str());
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    GLuint prog = makeProgram(damWvs.c_str(), damWfs.c_str());
    if (!prog) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    std::string solidVs = loadTextFile(shaderPath("dam_solid.vert"));
    std::string solidFs = loadTextFile(shaderPath("dam_solid.frag"));
    if (solidVs.empty() || solidFs.empty()) {
        std::fprintf(stderr, "failed to load dam solid shaders (tried %s)\n", shaderPath("dam_solid.vert").c_str());
        glDeleteProgram(prog);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    GLuint solidProg = makeProgram(solidVs.c_str(), solidFs.c_str());
    if (!solidProg) {
        glDeleteProgram(prog);
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    GLint locMVP = glGetUniformLocation(prog, "uMVP");
    GLint locLight = glGetUniformLocation(prog, "uLightDir");
    GLint locWaterAlpha = glGetUniformLocation(prog, "uAlpha");
    GLint locDx = glGetUniformLocation(prog, "uDx");
    GLint locHalfW = glGetUniformLocation(prog, "uHalfW");
    GLint locHalfD = glGetUniformLocation(prog, "uHalfD");
    GLint locTexH = glGetUniformLocation(prog, "uH");
    GLint locTexB = glGetUniformLocation(prog, "uB");
    GLint locWetEps = glGetUniformLocation(prog, "uWetDepthEps");
    GLint locSolidMVP = glGetUniformLocation(solidProg, "uMVP");
    GLint locSolidLight = glGetUniformLocation(solidProg, "uLightDir");
    GLint locSolidColor = glGetUniformLocation(solidProg, "uBaseColor");

    Grid g(kNx, kNy, kDx, kDt);
    setupDamInitialState(g);
    const float halfW = 0.5f * kNx * kDx;
    const float halfD = 0.5f * kNy * kDx;

    std::vector<unsigned> indices;
    buildGridIndices(kNx, kNy, indices);
    std::vector<float> waterCornerIJ;
    buildWaterCornerIJ(kNx, kNy, waterCornerIJ);
    std::vector<float> damWallVerts;
    buildDamWallMesh(damWallVerts, kNx, kNy, kDx, halfW, halfD);

    GLuint texH = 0, texB = 0;
    allocGridTextures(kNx, kNy, texH, texB);

    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(waterCornerIJ.size() * sizeof(float)), waterCornerIJ.data(),
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned)),
                 indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    GLuint wallVao = 0, wallVbo = 0;
    glGenVertexArrays(1, &wallVao);
    glGenBuffers(1, &wallVbo);
    glBindVertexArray(wallVao);
    glBindBuffer(GL_ARRAY_BUFFER, wallVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(damWallVerts.size() * sizeof(float)), damWallVerts.data(),
                 GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.12f, 0.16f, 0.24f, 1.f);

    glm::vec3 lightDir = glm::normalize(glm::vec3(0.35f, 0.85f, 0.4f));
    double fpsPrevT = glfwGetTime();
    float fpsShown = 0.f;
    double simT = 0.0;

    while (!glfwWindowShouldClose(window)) {
        const double now = glfwGetTime();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            setupDamInitialState(g);
            simT = 0.0;
        }

        for (int s = 0; s < kSubsteps; ++s) {
            sweStepGpu(g);
            simT += g.dt;
        }

        static double lastFroudePrintSimT = -1e9;
        constexpr double kFroudePrintInterval = 0.25;
        if (simT - lastFroudePrintSimT >= kFroudePrintInterval) {
            lastFroudePrintSimT = simT;
            const ShallowWaterDiagnostics diag = gridShallowWaterDiagnostics(g, kG);
            std::printf(
                "t=%.3f s  Fr_max=%.4f  |u|@Frmax=%.4f m/s  h@Frmax=%.4f m  |u|_max=%.4f  h_min_wet=%.4f m\n",
                simT, static_cast<double>(diag.fr_max), static_cast<double>(diag.speed_at_fr_max),
                static_cast<double>(diag.h_at_fr_max), static_cast<double>(diag.speed_max),
                static_cast<double>(diag.h_min_wet));
            std::fflush(stdout);
        }

        uploadGridTextures(g, texH, texB);

        const float aspect = frame.fbH > 0 ? static_cast<float>(frame.fbW) / static_cast<float>(frame.fbH) : 1.f;
        const float yaw = glm::radians(kFixedCamYawDeg);
        const float pitch = glm::radians(kFixedCamPitchDeg);
        glm::vec3 camFront(
            std::cos(yaw) * std::cos(pitch),
            std::sin(pitch),
            std::sin(yaw) * std::cos(pitch)
        );
        camFront = glm::normalize(camFront);
        const glm::vec3 worldUp(0.f, 1.f, 0.f);
        glm::vec3 camRight = glm::normalize(glm::cross(camFront, worldUp));
        glm::vec3 camUp = glm::normalize(glm::cross(camRight, camFront));
        glm::mat4 proj = glm::perspective(glm::radians(kFixedCamFovDeg), aspect, 0.1f, 1500.f);
        glm::mat4 view = glm::lookAt(kFixedCamPos, kFixedCamPos + camFront, camUp);
        glm::mat4 mvp = proj * view;

        glViewport(0, 0, frame.fbW, frame.fbH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
        glDisable(GL_CULL_FACE);
        glUseProgram(solidProg);
        glUniformMatrix4fv(locSolidMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3fv(locSolidLight, 1, glm::value_ptr(lightDir));
        const glm::vec3 wallColor(0.58f, 0.56f, 0.52f);
        glUniform3fv(locSolidColor, 1, glm::value_ptr(wallColor));
        glBindVertexArray(wallVao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(damWallVerts.size() / 6));
        glBindVertexArray(0);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        glDisable(GL_CULL_FACE);

        glUseProgram(prog);
        glUniformMatrix4fv(locMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3fv(locLight, 1, glm::value_ptr(lightDir));
        glUniform1f(locWaterAlpha, 0.9f);
        glUniform1f(locDx, kDx);
        glUniform1f(locHalfW, halfW);
        glUniform1f(locHalfD, halfD);
        // Same kWetDepthEps: vertex wet mask; fragment uses 2.5 * uWetDepthEps for depth fade (dam_water.frag).
        glUniform1f(locWetEps, kWetDepthEps);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texH);
        glUniform1i(locTexH, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texB);
        glUniform1i(locTexB, 1);
        glActiveTexture(GL_TEXTURE0);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        glfwSwapBuffers(window);
        glfwPollEvents();

        const double dtF = now - fpsPrevT;
        fpsPrevT = now;
        if (dtF > 1e-6 && dtF < 2.0) {
            const float inst = static_cast<float>(1.0 / dtF);
            fpsShown = (fpsShown < 1e-3f) ? inst : (fpsShown * 0.92f + inst * 0.08f);
        }
        char title[256];
        std::snprintf(title, sizeof title,
                      "Pure SWE Dam Break | %.0f FPS | t=%.2f s | fixed camera",
                      static_cast<double>(fpsShown), simT);
        glfwSetWindowTitle(window, title);
    }

    glDeleteProgram(prog);
    glDeleteProgram(solidProg);
    glDeleteTextures(1, &texH);
    glDeleteTextures(1, &texB);
    glDeleteBuffers(1, &wallVbo);
    glDeleteVertexArrays(1, &wallVao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
