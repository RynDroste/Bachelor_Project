#version 410 core
layout (location = 0) in vec2 aLocalXZ;

uniform mat4  uMVP;

// Clipmap level placement in world XZ
uniform vec2  uRingCenterXZ;
uniform float uRingSpacing;

// SWE window in world XZ: [center - half, center + half]
uniform vec2  uSweCenterXZ;
uniform vec2  uSweHalfExtent;

// Source fields (valid only inside the SWE window)
uniform sampler2D uH;
uniform sampler2D uB;
uniform float uWetDepthEps;
uniform float uEtaRef;
uniform float uShoreBlendRange;

// Gerstner surface aesthetic
uniform float uTime;
uniform float uGerstnerWeight;

out vec3  vWorldPos;
out float vDepth;
out float vGerstnerPeak;

const float PI = 3.14159265;
const float G  = 9.81;

void gerstnerAdd(vec2 xz, float t, vec2 D, float wavelength, float amp, float Q,
                 inout vec3 disp) {
    float k  = 2.0 * PI / max(wavelength, 0.001);
    float c  = sqrt(G / k);
    float a  = k * (dot(D, xz) - c * t);
    float s  = sin(a);
    float co = cos(a);
    disp.x += Q * amp * D.x * co;
    disp.z += Q * amp * D.y * co;
    disp.y += amp * s;
}

void main() {
    // Clipmap world position (XZ only; Y comes from SWE or rest level below)
    vec2 worldXZ = uRingCenterXZ + aLocalXZ * uRingSpacing;

    // Sample-point UV in the SWE source texture. Inside the SWE window we
    // average four cell samples to get a smooth surface height and depth.
    ivec2 texSz = textureSize(uH, 0);
    vec2  halfExt = max(uSweHalfExtent, vec2(1e-6));
    vec2  swePos = (worldXZ - uSweCenterXZ) / (2.0 * halfExt) + 0.5;

    float y     = uEtaRef;   // rest ocean level outside the SWE window
    float hAvg  = 0.0;

    // A few-texel padding so we don't sample garbage right at the border.
    vec2 uvMin = vec2(0.0);
    vec2 uvMax = vec2(1.0);
    bool inSwe = all(greaterThanEqual(swePos, uvMin)) &&
                 all(lessThanEqual(swePos,   uvMax));

    if (inSwe) {
        // Corresponding cell-center ivec2 in the SWE texture:
        // SWE cell (i,j) world center = uSweCenterXZ + ((i+0.5)/NX - 0.5) * 2*halfExt.
        // We average the nearest 2x2 cells around worldXZ for a smooth surface.
        vec2  cellf = swePos * vec2(texSz) - vec2(0.5);
        ivec2 c0    = ivec2(floor(cellf));
        vec2  f     = cellf - vec2(c0);

        float sumSurf = 0.0;
        float sumH    = 0.0;
        float sumW    = 0.0;
        float wLo = uWetDepthEps * 0.3;
        float wHi = uWetDepthEps * 1.6;
        for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) {
                int ci = clamp(c0.x + di, 0, texSz.x - 1);
                int cj = clamp(c0.y + dj, 0, texSz.y - 1);
                float wx = (di == 0) ? (1.0 - f.x) : f.x;
                float wz = (dj == 0) ? (1.0 - f.y) : f.y;
                float bw = wx * wz;

                float hv = texelFetch(uH, ivec2(ci, cj), 0).r;
                float bv = texelFetch(uB, ivec2(ci, cj), 0).r;
                float wetW = smoothstep(wLo, wHi, max(hv, 0.0));
                float w    = bw * wetW;
                sumSurf += w * (bv + hv);
                sumH    += w * hv;
                sumW    += w;
            }
        }

        if (sumW > 1e-6) {
            float ySampled = sumSurf / sumW;
            hAvg = sumH / sumW;
            float shoreBlend = uShoreBlendRange > 1e-6
                ? clamp(hAvg / uShoreBlendRange, 0.0, 1.0)
                : 1.0;
            y = mix(uEtaRef, ySampled, shoreBlend);
        }
    }

    // Gerstner aesthetic waves, applied everywhere (they read from world XZ).
    float gMask = clamp(uGerstnerWeight, 0.0, 1.0);
    if (inSwe) {
        float shoreBlend = uShoreBlendRange > 1e-6
            ? clamp(hAvg / uShoreBlendRange, 0.0, 1.0)
            : 1.0;
        float wetGerst = smoothstep(uWetDepthEps * 0.25, uWetDepthEps * 2.5, max(hAvg, 0.0));
        gMask *= mix(1.0, wetGerst * shoreBlend, 1.0);  // inside SWE: dampen on dry cells
    }

    vec3 gDisp = vec3(0.0);
    if (gMask > 1e-5) {
        vec2 xz = worldXZ;
        gerstnerAdd(xz, uTime, normalize(vec2(1.0, 0.22)), 42.0, 0.11, 0.55, gDisp);
        gerstnerAdd(xz, uTime, normalize(vec2(-0.35, 1.0)), 28.0, 0.065, 0.5, gDisp);
        gerstnerAdd(xz, uTime * 1.07, normalize(vec2(0.85, -0.52)), 16.0, 0.035, 0.42, gDisp);
        gerstnerAdd(xz, uTime * 0.93, normalize(vec2(0.2, 1.0)), 9.0, 0.018, 0.38, gDisp);
        gDisp *= gMask;
    }
    vGerstnerPeak = clamp(length(gDisp.xz) * 8.0, 0.0, 1.0);

    vWorldPos = vec3(worldXZ.x + gDisp.x, y + gDisp.y, worldXZ.y + gDisp.z);
    vDepth    = hAvg;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
