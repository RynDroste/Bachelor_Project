#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>
#include <vector>

namespace {

struct Vertex {
    glm::vec3 pos;
    glm::vec3 nrm;
};

struct FrameCtx {
    int w = 1280;
    int h = 720;
};

// Camera tuned to mimic the oblique top-down perspective in Fig.4.
constexpr float kCamFovDeg = 42.0f;
const glm::vec3 kCamPos(39.5f, 23.0f, -9.5f);
const glm::vec3 kCamTarget(3.5f, 0.8f, 0.0f);

static const char* kVs = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNrmW;
void main() {
    mat3 nrmMat = mat3(transpose(inverse(uModel)));
    vNrmW = normalize(nrmMat * aNrm);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)GLSL";

static const char* kFs = R"GLSL(
#version 330 core
in vec3 vNrmW;
uniform vec3 uColor;
uniform vec3 uLightDir;
out vec4 FragColor;
void main() {
    float ndl = max(dot(normalize(vNrmW), normalize(uLightDir)), 0.0);
    vec3 rgb = uColor * (0.25 + 0.75 * ndl);
    FragColor = vec4(rgb, 1.0);
}
)GLSL";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint makeProgram() {
    GLuint vs = compile(GL_VERTEX_SHADER, kVs);
    GLuint fs = compile(GL_FRAGMENT_SHADER, kFs);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
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
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

void pushTri(std::vector<Vertex>& v, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 n) {
    v.push_back({a, n});
    v.push_back({b, n});
    v.push_back({c, n});
}

void addBox(std::vector<Vertex>& out, glm::vec3 mn, glm::vec3 mx) {
    glm::vec3 p000{mn.x, mn.y, mn.z};
    glm::vec3 p001{mn.x, mn.y, mx.z};
    glm::vec3 p010{mn.x, mx.y, mn.z};
    glm::vec3 p011{mn.x, mx.y, mx.z};
    glm::vec3 p100{mx.x, mn.y, mn.z};
    glm::vec3 p101{mx.x, mn.y, mx.z};
    glm::vec3 p110{mx.x, mx.y, mn.z};
    glm::vec3 p111{mx.x, mx.y, mx.z};

    pushTri(out, p001, p101, p111, {0, 0, 1});
    pushTri(out, p001, p111, p011, {0, 0, 1});
    pushTri(out, p100, p000, p010, {0, 0, -1});
    pushTri(out, p100, p010, p110, {0, 0, -1});
    pushTri(out, p000, p001, p011, {-1, 0, 0});
    pushTri(out, p000, p011, p010, {-1, 0, 0});
    pushTri(out, p101, p100, p110, {1, 0, 0});
    pushTri(out, p101, p110, p111, {1, 0, 0});
    pushTri(out, p010, p011, p111, {0, 1, 0});
    pushTri(out, p010, p111, p110, {0, 1, 0});
    pushTri(out, p000, p100, p101, {0, -1, 0});
    pushTri(out, p000, p101, p001, {0, -1, 0});
}

void fbCB(GLFWwindow* w, int ww, int hh) {
    auto* ctx = static_cast<FrameCtx*>(glfwGetWindowUserPointer(w));
    if (ctx) {
        ctx->w = ww;
        ctx->h = hh;
    }
    glViewport(0, 0, ww, hh);
}

void drawMesh(GLuint p, GLuint vao, GLsizei n,
              const glm::mat4& vp, const glm::mat4& model,
              const glm::vec3& color, const glm::vec3& lightDir) {
    glUseProgram(p);
    glm::mat4 mvp = vp * model;
    glUniformMatrix4fv(glGetUniformLocation(p, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(glGetUniformLocation(p, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
    glUniform3fv(glGetUniformLocation(p, "uColor"), 1, glm::value_ptr(color));
    glUniform3fv(glGetUniformLocation(p, "uLightDir"), 1, glm::value_ptr(lightDir));
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, n);
}

} // namespace

int main() {
    if (!glfwInit()) {
        std::fprintf(stderr, "glfw init failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(1280, 720, "Dam Scene (3 Openings)", nullptr, nullptr);
    if (!win) {
        std::fprintf(stderr, "window create failed\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::fprintf(stderr, "glad load failed\n");
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }

    FrameCtx frame{};
    glfwSetWindowUserPointer(win, &frame);
    glfwSetFramebufferSizeCallback(win, fbCB);

    GLuint prog = makeProgram();
    if (!prog) {
        glfwDestroyWindow(win);
        glfwTerminate();
        return 1;
    }

    std::vector<Vertex> solid, water;

    // Ground slab
    addBox(solid, {-40.0f, -0.15f, -24.0f}, {40.0f, 0.0f, 24.0f});

    // Side walls (channel boundaries)
    addBox(solid, {-40.0f, 0.0f, -24.0f}, {40.0f, 2.2f, -22.0f});
    addBox(solid, {-40.0f, 0.0f, 22.0f}, {40.0f, 2.2f, 24.0f});

    // Dam wall at x = 0 with three openings
    constexpr float damX0 = -0.8f;
    constexpr float damX1 = 0.8f;
    constexpr float damH = 2.0f;
    addBox(solid, {damX0, 0.0f, -22.0f}, {damX1, damH, -12.0f}); // left segment
    addBox(solid, {damX0, 0.0f, -8.0f},  {damX1, damH, -2.0f});  // mid-left segment
    addBox(solid, {damX0, 0.0f, 2.0f},   {damX1, damH, 8.0f});   // mid-right segment
    addBox(solid, {damX0, 0.0f, 12.0f},  {damX1, damH, 22.0f});  // right segment
    // Openings are [-12,-8], [-2,2], [8,12] along z.

    // Upstream water block (dam-break initial water)
    addBox(water, {-38.0f, 0.0f, -22.0f}, {damX0, 1.2f, 22.0f});

    GLuint vaoSolid = 0, vboSolid = 0, vaoWater = 0, vboWater = 0;
    glGenVertexArrays(1, &vaoSolid);
    glGenBuffers(1, &vboSolid);
    glBindVertexArray(vaoSolid);
    glBindBuffer(GL_ARRAY_BUFFER, vboSolid);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(solid.size() * sizeof(Vertex)), solid.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nrm));

    glGenVertexArrays(1, &vaoWater);
    glGenBuffers(1, &vboWater);
    glBindVertexArray(vaoWater);
    glBindBuffer(GL_ARRAY_BUFFER, vboWater);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(water.size() * sizeof(Vertex)), water.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, pos));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, nrm));
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }

        glClearColor(0.84f, 0.80f, 0.72f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float aspect = (frame.h > 0) ? static_cast<float>(frame.w) / static_cast<float>(frame.h) : 1.0f;
        glm::mat4 proj = glm::perspective(glm::radians(kCamFovDeg), aspect, 0.1f, 300.0f);
        glm::mat4 view = glm::lookAt(kCamPos, kCamTarget, glm::vec3(0, 1, 0));
        glm::mat4 vp = proj * view;

        glm::vec3 lightDir = glm::normalize(glm::vec3(0.45f, 1.0f, 0.35f));
        drawMesh(prog, vaoSolid, static_cast<GLsizei>(solid.size()), vp, glm::mat4(1.0f),
                 glm::vec3(0.82f, 0.76f, 0.61f), lightDir);
        drawMesh(prog, vaoWater, static_cast<GLsizei>(water.size()), vp, glm::mat4(1.0f),
                 glm::vec3(0.42f, 0.70f, 0.90f), lightDir);

        glfwSwapBuffers(win);
    }

    glDeleteBuffers(1, &vboSolid);
    glDeleteVertexArrays(1, &vaoSolid);
    glDeleteBuffers(1, &vboWater);
    glDeleteVertexArrays(1, &vaoWater);
    glDeleteProgram(prog);
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

