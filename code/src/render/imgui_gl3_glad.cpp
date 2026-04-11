// Bridges GLAD (used by the app) with Dear ImGui's OpenGL3 backend without a second GL loader.
#include <glad/glad.h>
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#include "imgui_impl_opengl3.cpp"
