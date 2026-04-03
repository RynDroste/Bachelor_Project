#include "render/glfw_gl_window.h"

#include <GLFW/glfw3.h>
#include <cstdio>

namespace {

void hintCoreProfile(int major, int minor) {
    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
}

}  // namespace

GLFWwindow* glfwCreateWindowWithGlFallback(int width, int height, const char* title) {
    hintCoreProfile(4, 6);
    GLFWwindow* w = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (w) {
        return w;
    }

    std::fprintf(stderr, "OpenGL 4.6 core context unavailable; falling back to 3.3.\n");
    hintCoreProfile(3, 3);
    return glfwCreateWindow(width, height, title, nullptr, nullptr);
}
