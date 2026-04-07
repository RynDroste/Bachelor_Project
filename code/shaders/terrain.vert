#version 410 core
layout (location = 0) in vec2 aCornerIJ;
uniform mat4 uMVP;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform float uUvScale;
uniform sampler2D uB;
out vec3 vWorldPos;
out vec2 vIJ;
out vec2 vUv;
void main() {
    int vi = int(aCornerIJ.x + 0.0001);
    int vj = int(aCornerIJ.y + 0.0001);
    ivec2 sz = textureSize(uB, 0);
    int NX = sz.x;
    int NY = sz.y;
    float sumB = 0.0;
    int cnt = 0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ci = vi + di;
            int cj = vj + dj;
            if (ci >= 0 && ci < NX && cj >= 0 && cj < NY) {
                sumB += texelFetch(uB, ivec2(ci, cj), 0).r;
                cnt++;
            }
        }
    }
    float y = cnt > 0 ? sumB / float(cnt) : 0.0;
    float wx = float(vi) * uDx - uHalfW;
    float wz = float(vj) * uDx - uHalfD;
    vWorldPos = vec3(wx, y, wz);
    vIJ = aCornerIJ;
    vUv = vec2(wx, wz) * uUvScale;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
