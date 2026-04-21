#version 410 core
layout (location = 0) in vec2 aLocalXZ;

uniform mat4  uMVP;

// Clipmap level placement
uniform vec2  uRingCenterXZ;
uniform float uRingSpacing;

// SWE window
uniform vec2  uSweCenterXZ;
uniform vec2  uSweHalfExtent;

uniform sampler2D uB;
uniform float     uUvScale;

out vec3 vWorldPos;
out vec2 vUv;
out float vInSwe;

void main() {
    vec2 worldXZ = uRingCenterXZ + aLocalXZ * uRingSpacing;

    ivec2 texSz = textureSize(uB, 0);
    vec2  halfExt = max(uSweHalfExtent, vec2(1e-6));
    vec2  swePos = (worldXZ - uSweCenterXZ) / (2.0 * halfExt) + 0.5;
    bool  inSwe  = all(greaterThanEqual(swePos, vec2(0.0))) &&
                   all(lessThanEqual(swePos,   vec2(1.0)));

    float y = 0.0;  // flat riverbed default outside the SWE window
    if (inSwe) {
        // 2x2 bilinear over the 4 nearest cell centers (matches ocean.vert style).
        vec2  cellf = swePos * vec2(texSz) - vec2(0.5);
        ivec2 c0    = ivec2(floor(cellf));
        vec2  f     = cellf - vec2(c0);
        float sumB = 0.0, sumW = 0.0;
        for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) {
                int ci = clamp(c0.x + di, 0, texSz.x - 1);
                int cj = clamp(c0.y + dj, 0, texSz.y - 1);
                float wx = (di == 0) ? (1.0 - f.x) : f.x;
                float wz = (dj == 0) ? (1.0 - f.y) : f.y;
                float w  = wx * wz;
                sumB += w * texelFetch(uB, ivec2(ci, cj), 0).r;
                sumW += w;
            }
        }
        y = (sumW > 1e-6) ? (sumB / sumW) : 0.0;
    }

    vWorldPos = vec3(worldXZ.x, y, worldXZ.y);
    vUv       = worldXZ * uUvScale;
    vInSwe    = inSwe ? 1.0 : 0.0;
    gl_Position = uMVP * vec4(vWorldPos, 1.0);
}
