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

    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            const float x = static_cast<float>(j) / static_cast<float>(resolution - 1);
            const float y = static_cast<float>(i) / static_cast<float>(resolution - 1);

            // Background trend: higher terrain in the upper part, lower terrain in the lower part.
            const float upperPlateau = 0.58f * y;
            const float baseSlope = -0.22f * x + 0.04f * y;

            // Carve a broad transition valley between upper and lower regions.
            const float channelCenter =
                0.48f +
                0.05f * std::sin(2.0f * 3.14159265359f * (x + 0.08f)) -
                0.03f * std::sin(4.0f * 3.14159265359f * (x - 0.18f));
            const float valleyDist = (y - channelCenter) / 0.06f;
            const float valley = -0.28f * std::exp(-valleyDist * valleyDist);

            // Add roughness to mimic DEM-like ridges/erosion patterns.
            const float ridge1 = 0.06f * std::sin(9.0f * x + 3.0f * y);
            const float ridge2 = 0.04f * std::sin(18.0f * x - 11.0f * y);
            const float ridge3 = 0.02f * std::cos(32.0f * x + 21.0f * y);
            const float roughness = ridge1 + ridge2 + ridge3;

            // Small local mound on the upper region for visual variety.
            const float dxM = x - 0.68f;
            const float dyM = y - 0.78f;
            const float mound = 0.10f * std::exp(-(dxM * dxM / 0.02f + dyM * dyM / 0.015f));

            dem.elevation[static_cast<size_t>(i) * static_cast<size_t>(resolution) + static_cast<size_t>(j)] =
                upperPlateau + baseSlope + valley + roughness + mound;
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
    const bool forceProcedural = (argc > 1) && (std::string(argv[1]) == "--procedural");
    try {
        if (argc > 1 && !forceProcedural) {
            dem = loadDemFromFile(argv[1]);
            demLoaded = true;
        } else if (!forceProcedural) {
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
    const int injectionI = expectedResolution - 1;
    const int injectionJ = 0;

    std::vector<float> terrainVertices;
    std::vector<float> waterVertices;
    std::vector<unsigned int> indices;
    std::vector<unsigned int> wetWaterIndices;
    std::vector<float> waterBodyVertices;
    std::vector<unsigned int> waterBodyIndices;
    std::vector<unsigned char> wetCellMask;
    std::vector<unsigned char> wetVertexMask;
    terrainVertices.reserve((gridSize + 1) * (gridSize + 1) * 3);
    waterVertices.reserve((gridSize + 1) * (gridSize + 1) * 3);
    indices.reserve(gridSize * gridSize * 6);
    wetWaterIndices.reserve(indices.capacity());
    waterBodyVertices.reserve(gridSize * gridSize * 4 * 3);
    waterBodyIndices.reserve(gridSize * gridSize * 6);
    wetCellMask.resize(static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize), 0);
    wetVertexMask.resize(
        static_cast<size_t>(expectedResolution) * static_cast<size_t>(expectedResolution),
        0
    );

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
    solver.setInjection(
        injectionI,
        injectionJ,
        7.0f,
        0.05f,
        0.5f
    );
    const int N = solver.resolution();
    bool spaceWasPressed = false;

    float lastTime = static_cast<float>(glfwGetTime());
    GLuint terrainVAO, terrainVBO, waterVAO, waterVBO, waterBodyVAO, waterBodyVBO, terrainEBO, waterEBO, waterBodyEBO;
    glGenVertexArrays(1, &terrainVAO);
    glGenBuffers(1, &terrainVBO);
    glGenVertexArrays(1, &waterVAO);
    glGenBuffers(1, &waterVBO);
    glGenVertexArrays(1, &waterBodyVAO);
    glGenBuffers(1, &waterBodyVBO);
    glGenBuffers(1, &terrainEBO);
    glGenBuffers(1, &waterEBO);
    glGenBuffers(1, &waterBodyEBO);

    glBindVertexArray(terrainVAO);
    glBindBuffer(GL_ARRAY_BUFFER, terrainVBO);
    glBufferData(
        GL_ARRAY_BUFFER,
        terrainVertices.size() * sizeof(float),
        terrainVertices.data(),
        GL_STATIC_DRAW
    );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, terrainEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(waterVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
    glBufferData(GL_ARRAY_BUFFER, waterVertices.size() * sizeof(float), waterVertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(waterBodyVAO);
    glBindBuffer(GL_ARRAY_BUFFER, waterBodyVBO);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterBodyEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        const bool spaceIsPressed = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceIsPressed && !spaceWasPressed) {
            solver.setBathymetry(terrainOffset);
            solver.setInjection(
                injectionI,
                injectionJ,
                7.0f,
                0.005f,
                6.0f
            );
        }
        spaceWasPressed = spaceIsPressed;

        float currentTime = static_cast<float>(glfwGetTime());
        float frameDt = currentTime - lastTime;
        lastTime = currentTime;
        const float simulationTimeScale = 0.5f;
        solver.advance(frameDt * simulationTimeScale);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int v = (i * N + j) * 3;
                const float waterSurfaceY = solver.etaAt(i, j);
                waterVertices[v + 1] = waterSurfaceY;
            }
        }

        wetWaterIndices.clear();
        const float wetDepthOn = 3.0e-3f;
        const float wetDepthOff = 1.5e-3f;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const size_t id = static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j);
                const float depth = waterVertices[id * 3 + 1] - terrainVertices[id * 3 + 1];
                const bool previouslyWet = wetVertexMask[id] != 0;
                const bool nowWet = (depth >= wetDepthOn) || (previouslyWet && depth >= wetDepthOff);
                wetVertexMask[id] = nowWet ? 1 : 0;
            }
        }

        for (size_t k = 0; k + 2 < indices.size(); k += 3) {
            const unsigned int i0 = indices[k];
            const unsigned int i1 = indices[k + 1];
            const unsigned int i2 = indices[k + 2];

            const bool wet0 = wetVertexMask[static_cast<size_t>(i0)] != 0;
            const bool wet1 = wetVertexMask[static_cast<size_t>(i1)] != 0;
            const bool wet2 = wetVertexMask[static_cast<size_t>(i2)] != 0;
            if (wet0 || wet1 || wet2) {
                wetWaterIndices.push_back(i0);
                wetWaterIndices.push_back(i1);
                wetWaterIndices.push_back(i2);
            }
        }

        auto cellIndex = [gridSize](int i, int j) -> size_t {
            return static_cast<size_t>(i) * static_cast<size_t>(gridSize) + static_cast<size_t>(j);
        };
        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                const int id00 = i * N + j;
                const int id10 = i * N + (j + 1);
                const int id01 = (i + 1) * N + j;
                const int id11 = (i + 1) * N + (j + 1);
                const float d00 = waterVertices[static_cast<size_t>(id00) * 3 + 1] - terrainVertices[static_cast<size_t>(id00) * 3 + 1];
                const float d10 = waterVertices[static_cast<size_t>(id10) * 3 + 1] - terrainVertices[static_cast<size_t>(id10) * 3 + 1];
                const float d01 = waterVertices[static_cast<size_t>(id01) * 3 + 1] - terrainVertices[static_cast<size_t>(id01) * 3 + 1];
                const float d11 = waterVertices[static_cast<size_t>(id11) * 3 + 1] - terrainVertices[static_cast<size_t>(id11) * 3 + 1];
                const float maxDepth = std::max(std::max(d00, d10), std::max(d01, d11));
                const bool previouslyWet = wetCellMask[cellIndex(i, j)] != 0;
                const bool nowWet = (maxDepth >= wetDepthOn) || (previouslyWet && maxDepth >= wetDepthOff);
                wetCellMask[cellIndex(i, j)] = nowWet ? 1 : 0;
            }
        }

        waterBodyVertices.clear();
        waterBodyIndices.clear();
        auto pushVertex = [&waterBodyVertices](float x, float y, float z) -> unsigned int {
            const unsigned int idx = static_cast<unsigned int>(waterBodyVertices.size() / 3);
            waterBodyVertices.push_back(x);
            waterBodyVertices.push_back(y);
            waterBodyVertices.push_back(z);
            return idx;
        };
        auto addSideQuad = [&](int idA, int idB) {
            const float xA = terrainVertices[static_cast<size_t>(idA) * 3];
            const float zA = terrainVertices[static_cast<size_t>(idA) * 3 + 2];
            const float xB = terrainVertices[static_cast<size_t>(idB) * 3];
            const float zB = terrainVertices[static_cast<size_t>(idB) * 3 + 2];
            const float topA = waterVertices[static_cast<size_t>(idA) * 3 + 1];
            const float topB = waterVertices[static_cast<size_t>(idB) * 3 + 1];
            const float botA = terrainVertices[static_cast<size_t>(idA) * 3 + 1];
            const float botB = terrainVertices[static_cast<size_t>(idB) * 3 + 1];

            const unsigned int v0 = pushVertex(xA, topA, zA);
            const unsigned int v1 = pushVertex(xB, topB, zB);
            const unsigned int v2 = pushVertex(xA, botA, zA);
            const unsigned int v3 = pushVertex(xB, botB, zB);

            waterBodyIndices.push_back(v0);
            waterBodyIndices.push_back(v2);
            waterBodyIndices.push_back(v1);
            waterBodyIndices.push_back(v1);
            waterBodyIndices.push_back(v2);
            waterBodyIndices.push_back(v3);
        };

        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                if (!wetCellMask[cellIndex(i, j)]) {
                    continue;
                }

                const int id00 = i * N + j;
                const int id10 = i * N + (j + 1);
                const int id01 = (i + 1) * N + j;
                const int id11 = (i + 1) * N + (j + 1);

                const bool northDry = (i == 0) || (wetCellMask[cellIndex(i - 1, j)] == 0);
                const bool southDry = (i == gridSize - 1) || (wetCellMask[cellIndex(i + 1, j)] == 0);
                const bool westDry = (j == 0) || (wetCellMask[cellIndex(i, j - 1)] == 0);
                const bool eastDry = (j == gridSize - 1) || (wetCellMask[cellIndex(i, j + 1)] == 0);

                if (northDry) {
                    addSideQuad(id00, id10);
                }
                if (southDry) {
                    addSideQuad(id01, id11);
                }
                if (westDry) {
                    addSideQuad(id00, id01);
                }
                if (eastDry) {
                    addSideQuad(id10, id11);
                }
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, waterVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, waterVertices.size() * sizeof(float), waterVertices.data());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterEBO);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            wetWaterIndices.size() * sizeof(unsigned int),
            wetWaterIndices.data(),
            GL_DYNAMIC_DRAW
        );
        glBindBuffer(GL_ARRAY_BUFFER, waterBodyVBO);
        glBufferData(
            GL_ARRAY_BUFFER,
            waterBodyVertices.size() * sizeof(float),
            waterBodyVertices.data(),
            GL_DYNAMIC_DRAW
        );
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, waterBodyEBO);
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            waterBodyIndices.size() * sizeof(unsigned int),
            waterBodyIndices.data(),
            GL_DYNAMIC_DRAW
        );

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
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(wetWaterIndices.size()), GL_UNSIGNED_INT, 0);

        glUniform3f(colorLocation, 0.12f, 0.40f, 0.95f);
        glUniform1f(alphaLocation, 0.55f);
        glBindVertexArray(waterBodyVAO);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(waterBodyIndices.size()), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
