#version 410 core
layout (location = 0) in vec2 aCornerIJ;
uniform mat4 uMVP;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform sampler2D uH;
uniform sampler2D uB;
uniform float uWetDepthEps;
out vec3 vWorldPos;
out float vDepth;
out float vWetFrac;
void main() {
    int vi = int(aCornerIJ.x + 0.0001);
    int vj = int(aCornerIJ.y + 0.0001);
    ivec2 sz = textureSize(uH, 0);
    int NX = sz.x;
    int NY = sz.y;
    float sumSurf = 0.0;
    float sumH = 0.0;
    float sumW = 0.0;
    float wLo = uWetDepthEps * 0.3;
    float wHi = uWetDepthEps * 1.6;
    for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
            int ci = vi + di;
            int cj = vj + dj;
            if (ci >= 0 && ci < NX && cj >= 0 && cj < NY) {
                float hv = texelFetch(uH, ivec2(ci, cj), 0).r;
                float bv = texelFetch(uB, ivec2(ci, cj), 0).r;
                float w = smoothstep(wLo, wHi, max(hv, 0.0));
                sumSurf += w * (bv + hv);
                sumH += w * hv;
                sumW += w;
            }
        }
    }
    float y = 0.0;
    float hAvg = 0.0;
    if (sumW > 1e-6) {
        y = sumSurf / sumW;
        hAvg = sumH / sumW;
    } else {
        float sumBed = 0.0;
        int cntBed = 0;
        for (int di = -1; di <= 0; ++di) {
            for (int dj = -1; dj <= 0; ++dj) {
                int ci = vi + di;
                int cj = vj + dj;
                if (ci >= 0 && ci < NX && cj >= 0 && cj < NY) {
                    sumBed += texelFetch(uB, ivec2(ci, cj), 0).r;
                    cntBed++;
                }
            }
        }
        y = (cntBed > 0) ? (sumBed / float(cntBed)) : 0.0;
    }
    vWetFrac = sumW * 0.25;
    float wx = float(vi) * uDx - uHalfW;
    float wz = float(vj) * uDx - uHalfD;
    vWorldPos = vec3(wx, y, wz);
    vDepth = hAvg;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
