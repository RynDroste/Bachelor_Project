#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
uniform mat4 uMVP;
uniform int uClipRefl;
uniform float uWaterPlaneY;
// uClipSide: +1 keep y >= waterY (above-water reflection),
//            -1 keep y <= waterY (underwater reflection).
uniform int uClipSide;
out vec3 vNrm;
out vec3 vWorldPos;
void main() {
    vNrm = aNrm;
    vWorldPos = aPos;
    if (uClipRefl != 0) {
        float side = (uClipSide < 0) ? -1.0 : 1.0;
        gl_ClipDistance[0] = side * (aPos.y - uWaterPlaneY);
    } else {
        gl_ClipDistance[0] = 1.0;
    }
    gl_Position = uMVP * vec4(aPos, 1.0);
}
