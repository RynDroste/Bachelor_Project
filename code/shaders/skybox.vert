#version 410 core

layout(location = 0) in vec3 aPos;

out vec3 vDir;

uniform mat4 uViewProj;

void main() {
    vDir = aPos;
    vec4 clip = uViewProj * vec4(aPos, 1.0);
    gl_Position = clip.xyww;
}
