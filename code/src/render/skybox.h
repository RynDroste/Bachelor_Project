#pragma once

#include <glad/glad.h>
#include <glm/mat4x4.hpp>

struct SkyboxGL {
    GLuint cubemap = 0;
    GLuint vao     = 0;
    GLuint vbo     = 0;
    GLuint program = 0;
    GLint  locViewProj = -1;
    GLint  locSky      = -1;
};

bool skyboxInit(SkyboxGL& out, const char* skyboxRootDir);
void skyboxShutdown(SkyboxGL& sb);
void skyboxDraw(const SkyboxGL& sb, const glm::mat4& proj, const glm::mat4& view);
