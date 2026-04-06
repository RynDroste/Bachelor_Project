#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
uniform mat4 uMVP;
uniform int uClipRefl;
uniform float uWaterPlaneY;
out vec3 vNrm;
void main() {
    vNrm = aNrm;
    if (uClipRefl != 0)
        gl_ClipDistance[0] = aPos.y - uWaterPlaneY;
    else
        gl_ClipDistance[0] = 1.0;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
