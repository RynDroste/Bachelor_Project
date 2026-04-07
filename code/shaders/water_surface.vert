#version 410 core
// Grid corners: blend wet/dry; Gerstner is visual-only (not SWE).
layout (location = 0) in vec2 aCornerIJ;
uniform mat4 uMVP;
uniform mat4 uLightSpace;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform sampler2D uH;
uniform sampler2D uB;
uniform float uWetDepthEps;
uniform float uEtaRef;
uniform float uShoreBlendRange;
uniform float uTime;
uniform float uGerstnerWeight;
out vec3 vWorldPos;
out vec4 vLightSpacePos;
out float vDepth;

const float PI = 3.14159265;
const float G = 9.81;

void gerstnerAdd(vec2 xz, float t, vec2 D, float wavelength, float amp, float Q,
                 inout vec3 disp) {
    float k = 2.0 * PI / max(wavelength, 0.001);
    float c = sqrt(G / k);
    float a = k * (dot(D, xz) - c * t);
    float s = sin(a);
    float co = cos(a);
    disp.x += Q * amp * D.x * co;
    disp.z += Q * amp * D.y * co;
    disp.y += amp * s;
}

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
    float y;
    float hAvg;
    if (sumW > 1e-6) {
        float ySampled = sumSurf / sumW;
        hAvg = sumH / sumW;
        float shoreBlend = uShoreBlendRange > 1e-6
            ? clamp(hAvg / uShoreBlendRange, 0.0, 1.0)
            : 1.0;
        y = mix(uEtaRef, ySampled, shoreBlend);
    } else {
        y = uEtaRef;
        hAvg = 0.0;
    }
    float wx = float(vi) * uDx - uHalfW;
    float wz = float(vj) * uDx - uHalfD;

    float shoreBlend = 0.0;
    if (sumW > 1e-6) {
        shoreBlend = uShoreBlendRange > 1e-6
            ? clamp(hAvg / uShoreBlendRange, 0.0, 1.0)
            : 1.0;
    }
    float wetGerst = smoothstep(uWetDepthEps * 0.25, uWetDepthEps * 2.5, max(hAvg, 0.0));
    float gMask = wetGerst * shoreBlend * clamp(uGerstnerWeight, 0.0, 1.0);

    vec3 gDisp = vec3(0.0);
    if (gMask > 1e-5) {
        vec2 xz = vec2(wx, wz);
        gerstnerAdd(xz, uTime, normalize(vec2(1.0, 0.22)), 42.0, 0.11, 0.55, gDisp);
        gerstnerAdd(xz, uTime, normalize(vec2(-0.35, 1.0)), 28.0, 0.065, 0.5, gDisp);
        gerstnerAdd(xz, uTime * 1.07, normalize(vec2(0.85, -0.52)), 16.0, 0.035, 0.42, gDisp);
        gerstnerAdd(xz, uTime * 0.93, normalize(vec2(0.2, 1.0)), 9.0, 0.018, 0.38, gDisp);
        gDisp *= gMask;
    }

    vWorldPos = vec3(wx + gDisp.x, y + gDisp.y, wz + gDisp.z);
    vLightSpacePos = uLightSpace * vec4(vWorldPos, 1.0);
    vDepth = hAvg;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
