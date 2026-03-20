#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shallow_water_solver.h"
#include "terrain_loader.h"

namespace {

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    glViewport(0, 0, width, height);
}

DemData buildProceduralDem(int resolution) {
    DemData dem;
    dem.width = resolution;
    dem.height = resolution;
    dem.originX = 0.0;
    dem.originY = 0.0;
    dem.pixelSizeX = 1.0;
    dem.pixelSizeY = 1.0;
    dem.hasNoData = false;
    dem.noDataValue = 0.0;
    dem.elevation.resize(static_cast<size_t>(resolution) * static_cast<size_t>(resolution), 0.0f);

    const float center = 0.5f * static_cast<float>(resolution - 1);
    const float sigma = 0.18f * static_cast<float>(resolution);

    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            const float x = static_cast<float>(j);
            const float y = static_cast<float>(i);
            const float dx = x - center;
            const float dy = y - center;
            const float r2 = dx * dx + dy * dy;

            const float hill = 0.28f * std::exp(-r2 / (2.0f * sigma * sigma));
            const float slope = 0.0020f * (x - center);
            const float ripple =
                0.03f * std::sin(2.0f * 3.14159265359f * x / static_cast<float>(resolution)) *
                std::sin(2.0f * 3.14159265359f * y / static_cast<float>(resolution));

            dem.elevation[static_cast<size_t>(i) * static_cast<size_t>(resolution) + static_cast<size_t>(j)] =
                hill + slope + ripple;
        }
    }

    return dem;
}

std::vector<float> buildTerrainOffset(const DemData& dem, float amplitude) {
    const size_t count = dem.elevation.size();
    std::vector<float> offset(count, 0.0f);
    if (count == 0) {
        return offset;
    }

    float minZ = dem.elevation[0];
    float maxZ = dem.elevation[0];
    for (float value : dem.elevation) {
        minZ = std::min(minZ, value);
        maxZ = std::max(maxZ, value);
    }

    const float range = std::max(maxZ - minZ, 1e-6f);
    for (size_t i = 0; i < count; ++i) {
        const float normalized = (dem.elevation[i] - minZ) / range;
        offset[i] = (normalized - 0.5f) * amplitude;
    }
    return offset;
}

}  // namespace

const GLchar* vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 position;\n"
    "uniform mat4 uMVP;\n"
    "void main()\n"
    "{\n"
    "gl_Position = uMVP * vec4(position, 1.0);\n"
    "}\0";

const GLchar* fragmentShaderSource = "#version 330 core\n"
    "uniform vec3 uColor;\n"
    "uniform float uAlpha;\n"
    "out vec4 color;\n"
    "void main()\n"
    "{\n"
    "color = vec4(uColor, uAlpha);\n"
    "}\n\0";  //ShaderSource in GLSL

int main(int argc, char** argv) {
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

    const GLint colorLocation = glGetUniformLocation(shaderProgram, "uColor");
    const GLint alphaLocation = glGetUniformLocation(shaderProgram, "uAlpha");
    const GLint mvpLocation = glGetUniformLocation(shaderProgram, "uMVP");

    const int gridSize = 128;
    const int expectedResolution = gridSize + 1;
    const std::array<std::string, 3> defaultDemCandidates = {
        "src/dem_129_utm.asc",
        "../src/dem_129_utm.asc",
        "dem_129_utm.asc"
    };

    DemData dem;
    bool demLoaded = false;
    try {
        if (argc > 1) {
            dem = loadDemFromFile(argv[1]);
            demLoaded = true;
        } else {
            for (const std::string& candidate : defaultDemCandidates) {
                try {
                    dem = loadDemFromFile(candidate);
                    demLoaded = true;
                    break;
                } catch (const std::exception&) {
                    // Try next candidate.
                }
            }
        }
    } catch (const std::exception&) {
    }

    if (!demLoaded) {
        dem = buildProceduralDem(expectedResolution);
        demLoaded = true;
    }

    if (dem.width != expectedResolution || dem.height != expectedResolution) {
        dem = buildProceduralDem(expectedResolution);
    }

    const float dxMeters = static_cast<float>(std::fabs(dem.pixelSizeX));
    const float dyMeters = static_cast<float>(std::fabs(dem.pixelSizeY));
    const float domainWidthMeters = dxMeters * static_cast<float>(gridSize);
    const float domainHeightMeters = dyMeters * static_cast<float>(gridSize);
    const float maxDomainMeters = std::max(domainWidthMeters, domainHeightMeters);
    if (maxDomainMeters <= 1e-6f) {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Non-dimensionalize coordinates to keep a stable simulation scale while preserving aspect ratio.
    // We map the larger physical domain extent to 2 model units (roughly the previous [-1, 1] range).
    const float modelUnitsPerMeter = 2.0f / maxDomainMeters;
    const float dxModel = dxMeters * modelUnitsPerMeter;
    const float dyModel = dyMeters * modelUnitsPerMeter;
    const float domainWidthModel = domainWidthMeters * modelUnitsPerMeter;
    const float domainHeightModel = domainHeightMeters * modelUnitsPerMeter;
    const std::vector<float> terrainOffset = buildTerrainOffset(dem, 0.15f);

    std::vector<float> terrainVertices;
    std::vector<float> waterVertices;
    std::vector<unsigned int> indices;
    terrainVertices.reserve((gridSize + 1) * (gridSize + 1) * 3);
    waterVertices.reserve((gridSize + 1) * (gridSize + 1) * 3);
    indices.reserve(gridSize * gridSize * 6);

    for (int i = 0; i <= gridSize; i++) {
        for (int j = 0; j <= gridSize; j++) {
            const float x = (static_cast<float>(j) / static_cast<float>(gridSize) - 0.5f) * domainWidthModel;
            const float z = (static_cast<float>(i) / static_cast<float>(gridSize) - 0.5f) * domainHeightModel;
            const float terrainY =
                terrainOffset[static_cast<size_t>(i) * static_cast<size_t>(expectedResolution) + static_cast<size_t>(j)];

            terrainVertices.push_back(x);
            terrainVertices.push_back(terrainY);
            terrainVertices.push_back(z);

            waterVertices.push_back(x);
            waterVertices.push_back(terrainY);
            waterVertices.push_back(z);
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

    ShallowWaterSolver solver(gridSize, dxModel, dyModel);
    solver.setBathymetry(terrainOffset);
    const int N = solver.resolution();

    float lastTime = static_cast<float>(glfwGetTime());
    GLuint terrainVAO, terrainVBO, waterVAO, waterVBO, EBO;
    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenVertexArrays(1, &waterVAO);
    glGenBuffers(1, &waterVBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(terrainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        terrainVertices.size() * sizeof(float),
        terrainVertices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(waterVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferData(GL_ARRAY_BUFFER, waterVertices.size() * sizeof(float), waterVertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }

        float currentTime = static_cast<float>(glfwGetTime());
        float frameDt = currentTime - lastTime;
        lastTime = currentTime;
        const float simulationTimeScale = 0.5f;
        solver.advance(frameDt * simulationTimeScale);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int v = (i * N + j) * 3;
                waterVertices[v + 1] =
                    terrainOffset[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)] +
                    solver.etaAt(i, j);
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, waterVertices.size() * sizeof(float), waterVertices.data());

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        int framebufferWidth = 800;
        int framebufferHeight = 600;
        glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
        const float aspect =
            static_cast<float>(std::max(framebufferWidth, 1)) / static_cast<float>(std::max(framebufferHeight, 1));

        // Oblique perspective camera similar to a 3/4 terrain overview.
        const glm::mat4 model = glm::mat4(1.0f);
        const glm::mat4 view = glm::lookAt(
            glm::vec3(1.9f, 1.35f, 1.9f),
            glm::vec3(0.0f, -0.05f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        const glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 10.0f);
        const glm::mat4 mvp = projection * view * model;
        glUniformMatrix4fv(mvpLocation, 1, GL_FALSE, glm::value_ptr(mvp));

        // Draw terrain in yellow.
        glUniform3f(colorLocation, 0.82f, 0.72f, 0.35f);
        glUniform1f(alphaLocation, 1.0f);
        glBindVertexArray(terrainVAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);

        // Draw water in blue.
        glUniform3f(colorLocation, 0.12f, 0.40f, 0.95f);
        glUniform1f(alphaLocation, 0.75f);
        glBindVertexArray(waterVAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
