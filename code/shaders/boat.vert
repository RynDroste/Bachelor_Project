#version 410 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNrm;
layout (location = 2) in vec2 aUV;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform int uClipRefl;
uniform float uWaterPlaneY;
// uClipSide: +1 keep y >= waterY (above-water reflection),
//            -1 keep y <= waterY (underwater reflection).
uniform int uClipSide;
out vec3 vNrm;
out vec3 vWorldPos;
out vec2 vUV;
void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    // Uniform scale only — mat3(uModel) is sufficient for normal transform.
    vNrm = mat3(uModel) * aNrm;
    vUV = aUV;
    if (uClipRefl != 0) {
        float side = (uClipSide < 0) ? -1.0 : 1.0;
        gl_ClipDistance[0] = side * (worldPos.y - uWaterPlaneY);
    } else {
        gl_ClipDistance[0] = 1.0;
    }
    gl_Position = uMVP * worldPos;
}
