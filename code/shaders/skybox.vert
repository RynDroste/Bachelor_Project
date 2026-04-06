#version 410 core
layout (location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 uProj;
uniform mat4 uViewSky;
void main() {
    TexCoords = aPos;
    vec4 clip = uProj * uViewSky * vec4(aPos, 1.0);
    gl_Position = clip.xyww;
}
