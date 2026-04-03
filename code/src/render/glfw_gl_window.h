#pragma once

struct GLFWwindow;

// Prefer OpenGL 4.6 Core; if the driver rejects it, fall back to 3.3 Core.
GLFWwindow* glfwCreateWindowWithGlFallback(int width, int height, const char* title);
