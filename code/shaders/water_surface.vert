#version 330 core
// Corner IJ; H/B from textures.
layout (location = 0) in vec2 aCornerIJ;
uniform mat4 uMVP;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform sampler2D uH;
uniform sampler2D uB;
out vec3 vWorldPos;
out float vDepth;
void main() {
    int vi = int(aCornerIJ.x + 0.0001);
    int vj = int(aCornerIJ.y + 0.0001);
    ivec2 sz = textureSize(uH, 0);
    int NX = sz.x;
    int NY = sz.y;
    float sumSurf = 0.0;
    float sumH = 0.0;
    int cntSurf = 0;
    int cntH = 0;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ci = vi + di;
            int cj = vj + dj;
            if (ci >= 0 && ci < NX && cj >= 0 && cj < NY) {
                float hv = texelFetch(uH, ivec2(ci, cj), 0).r;
                float bv = texelFetch(uB, ivec2(ci, cj), 0).r;
                sumSurf += bv + hv;
                sumH += hv;
                cntSurf++;
                cntH++;
            }
        }
    }
    float y = cntSurf > 0 ? sumSurf / float(cntSurf) : 0.0;
    float hAvg = cntH > 0 ? sumH / float(cntH) : 0.0;
    float wx = float(vi) * uDx - uHalfW;
    float wz = float(vj) * uDx - uHalfD;
    vWorldPos = vec3(wx, y, wz);
    vDepth = hAvg;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
