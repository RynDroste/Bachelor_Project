#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE

#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace {

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);
}

}  // namespace

const GLchar* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n"
    "void main()\n"
    "{\n"
    "gl_Position = vec4(position.x, position.y - position.z * 0.5, position.z, 1.0);\n"
    "}\0";

const GLchar* fragmentShaderSource = "#version 330 core\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
    "}\n\0";  //ShaderSource in GLSL

int main() {
    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__ 
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(800, 600, "BachelorProject", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cout << "ERROR::PROGRAM::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    const int  gridSize = 128;
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    vertices.reserve((gridSize + 1) * (gridSize + 1) * 3);
    indices.reserve(gridSize * gridSize * 6);

    for (int i = 0; i <= gridSize; i++) {
        for (int j = 0; j <= gridSize; j++) {
            float x = static_cast<float>(j) / static_cast<float>(gridSize) * 2.0f - 1.0f;
            float z = static_cast<float>(i) / static_cast<float>(gridSize) * 2.0f - 1.0f;
            float y = 0.0f; 
            
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridSize; ++j) {
            unsigned int topLeft = i * (gridSize + 1) + j;
            unsigned int topRight = topLeft + 1;
            unsigned int bottomLeft = (i + 1) * (gridSize + 1) + j;
            unsigned int bottomRight = bottomLeft + 1;

            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    const int N = gridSize + 1;
    const float dx = 2.0f / gridSize;
    const float a = 0.5f;
    const float dt = 1.0f / 120.0f; //Assume 120 fps;
    const float c = dx / dt * a; //wave speed
    const float damping = 0.999f;

    std::vector<float> hPrev(N * N, 0.0f);
    std::vector<float> hCurr(N * N, 0.0f);
    std::vector<float> hNext(N * N, 0.0f);

    auto idx = [N](int i, int j) { return i * N + j; };

    hCurr[idx(N/2, N/2)] = 0.5f; //Initial disturbance
    hPrev = hCurr;
    const float lambda = (c * dt / dx) * (c * dt / dx);
    if (lambda > a) {
        std::cout << "Warning: CFL may be unstable (lambda = " << lambda << ")\n";
    }

    float lastTime = static_cast<float>(glfwGetTime());
    float accumulator = 0.0f;
    GLuint VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    
    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        float currentTime = static_cast<float>(glfwGetTime());
        float frameDt = currentTime - lastTime;
        lastTime = currentTime;
        if (frameDt > 0.1f) {
            frameDt = 0.1f;
        }
        accumulator += frameDt;

        while (accumulator >= dt) {
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    float hCenter = hCurr[idx(i, j)];
                    float laplacian =
                        hCurr[idx(i, j - 1)] + hCurr[idx(i, j + 1)] +
                        hCurr[idx(i - 1, j)] + hCurr[idx(i + 1, j)] -
                        4.0f * hCenter;

                    hNext[idx(i, j)] =
                        (2.0f * hCenter - hPrev[idx(i, j)] + lambda * laplacian) * damping;
                }
            }

            for (int i = 0; i < N; ++i) {
                hNext[idx(i, 0)] = 0.0f;
                hNext[idx(i, N - 1)] = 0.0f;
            }
            for (int j = 0; j < N; ++j) {
                hNext[idx(0, j)] = 0.0f;
                hNext[idx(N - 1, j)] = 0.0f;
            }

            hPrev.swap(hCurr);
            hCurr.swap(hNext);
            accumulator -= dt;
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int v = (i * N + j) * 3;
                vertices[v + 1] = hCurr[idx(i, j)];
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
