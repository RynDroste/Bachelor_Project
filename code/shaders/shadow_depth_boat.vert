#version 410 core
layout (location = 0) in vec3 aPos;
uniform mat4 uLightSpace;
void main() {
    gl_Position = uLightSpace * vec4(aPos, 1.0);
}
