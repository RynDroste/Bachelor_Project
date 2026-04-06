#version 330 core
in vec3 vWorldPos;
in float vDepth;
in vec4 vClipPos;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform samplerCube uEnvMap;
uniform float uEnvMaxMip;
uniform sampler2D uReflectionTex;
uniform mat4 uReflViewProj;
uniform float uAlpha;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform sampler2D uH;
uniform float uWetDepthEps;
// Procedural bump (fBM + finite differences); matches wall-clock uTime from main.
uniform float uTime;
uniform float uWaveScale;
uniform float uWaveStrength;
uniform float uWaterAnimation;
uniform sampler2D uRefractionTex;
uniform sampler2D uRefractionDepth;
uniform float uZNear;
uniform float uZFar;
uniform float uDistortStrength;
uniform float uIOR;
uniform float uWaterReflectionsMix;
uniform vec3 uAbsorptionColor;
uniform float uAbsorptionDensity;
uniform float uTransparency;
uniform vec3 uEmissionColor;
uniform float uEmissionStrength;
out vec4 FragColor;

const float PI = 3.14159265;
// Perceptual roughness [0,1]; shared by Hammon diffuse and GGX specular (α = roughness²).
const float kRoughness = 0.045;
// Air–water interface, normal-incidence reflectance (~IOR 1.33).
const vec3 kF0 = vec3(0.02);
// IBL: cubemap mips approximate prefilter; no BRDF LUT (split-sum) — strengths are artistic knobs.
const float kIBLDiffuseMul = 0.62;
const float kIBLSpecMul    = 0.62;
// Planar reflection vs cubemap spec: 1 = full replace of IBL spec where valid.
const float kPlanarReflMix = 0.82;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float nh = max(dot(N, H), 0.0);
    float nh2 = nh * nh;
    float denom = nh2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float nx, float k) {
    return nx / (nx * (1.0 - k) + k);
}

float geometrySmith(float nv, float nl, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return geometrySchlickGGX(nv, k) * geometrySchlickGGX(nl, k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    return F0 + (1.0 - F0) * (t * t * t * t * t);
}

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

// Earl Hammon–style diffuse for GGX+Smith microsurfaces (single + multi scatter fit).
// roughness: perceptual [0,1]; same parameterization as kRoughness below for specular.
float hash11(float n) {
    return fract(sin(n) * 43758.5453123);
}

// [0,1] trilinear value noise; p in continuous space.
float noise3(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash11(dot(i, vec3(1.0, 57.0, 113.0)));
    float n100 = hash11(dot(i + vec3(1.0, 0.0, 0.0), vec3(1.0, 57.0, 113.0)));
    float n010 = hash11(dot(i + vec3(0.0, 1.0, 0.0), vec3(1.0, 57.0, 113.0)));
    float n110 = hash11(dot(i + vec3(1.0, 1.0, 0.0), vec3(1.0, 57.0, 113.0)));
    float n001 = hash11(dot(i + vec3(0.0, 0.0, 1.0), vec3(1.0, 57.0, 113.0)));
    float n101 = hash11(dot(i + vec3(1.0, 0.0, 1.0), vec3(1.0, 57.0, 113.0)));
    float n011 = hash11(dot(i + vec3(0.0, 1.0, 1.0), vec3(1.0, 57.0, 113.0)));
    float n111 = hash11(dot(i + vec3(1.0, 1.0, 1.0), vec3(1.0, 57.0, 113.0)));
    float nx0 = mix(n000, n100, f.x);
    float nx1 = mix(n010, n110, f.x);
    float nx2 = mix(n001, n101, f.x);
    float nx3 = mix(n011, n111, f.x);
    float ny0 = mix(nx0, nx1, f.y);
    float ny1 = mix(nx2, nx3, f.y);
    return mix(ny0, ny1, f.z);
}

// Smooth along w by blending two 3D noises (4th dimension / time axis).
float noise4D(vec4 x) {
    float wi = floor(x.w);
    float wf = fract(x.w);
    wf = wf * wf * (3.0 - 2.0 * wf);
    vec3 p = x.xyz;
    float a = noise3(p + wi * vec3(17.13, 31.37, 11.71));
    float b = noise3(p + (wi + 1.0) * vec3(17.13, 31.37, 11.71));
    return mix(a, b, wf);
}

// 3 octaves, lacunarity 2, amplitudes 0.5, 0.25, 0.125; sample in [-1,1] range.
float fbm4Raw(vec4 p) {
    float v = 0.0;
    float a = 0.5;
    vec4 f = p;
    for (int i = 0; i < 3; ++i) {
        v += a * (noise4D(f) * 2.0 - 1.0);
        f *= 2.0;
        a *= 0.5;
    }
    return v;
}

// Object/world-style mapping: Y scaled ×2 like the reference graph; W animates in w.
vec4 waterWaveCoord(vec3 worldPos, float W) {
    float s = uWaveScale;
    return vec4(worldPos.x * 1.0 * s, worldPos.y * 2.0 * s, worldPos.z * 1.0 * s, W);
}

// Finite differences in mapped (x,z): robust when N ≈ up (avoids degenerate ∂/∂y).
vec3 waterWavesNormal(vec3 worldPos, vec3 Ngeom, float W) {
    if (uWaveStrength <= 1e-6)
        return Ngeom;
    const float eps = 0.001;
    vec4 c0 = waterWaveCoord(worldPos, W);
    float h0 = fbm4Raw(c0);
    float hx = fbm4Raw(c0 + vec4(eps, 0.0, 0.0, 0.0));
    float hz = fbm4Raw(c0 + vec4(0.0, 0.0, eps, 0.0));
    float gx = (hx - h0) / eps;
    float gz = (hz - h0) / eps;

    vec3 Tn = vec3(1.0, 0.0, 0.0) - Ngeom * Ngeom.x;
    if (dot(Tn, Tn) < 1e-8)
        Tn = vec3(0.0, 0.0, 1.0) - Ngeom * Ngeom.z;
    Tn = normalize(Tn);
    vec3 Bn = normalize(cross(Ngeom, Tn));
    return normalize(Ngeom - uWaveStrength * (gx * Tn + gz * Bn));
}

float linearizeDepthBuf(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * uZNear * uZFar) / (uZFar + uZNear - z * (uZFar - uZNear));
}

vec3 hammonDiffuse(vec3 N, vec3 L, vec3 V, vec3 albedo, float roughness) {
    float NoL = clamp(dot(N, L), 0.0, 1.0);
    float NoV = clamp(dot(N, V), 0.0, 1.0);
    float LoV = dot(L, V);

    float a  = roughness * roughness;
    float a2 = a * a;

    float facing = 0.5 + 0.5 * LoV;

    float smoothTerm = (1.0 - pow5(1.0 - NoL)) * (1.0 - pow5(1.0 - NoV));
    smoothTerm *= 1.05;

    float NoH  = clamp(0.5 + 0.5 * NoL, 0.0, 1.0);
    float roughTerm = facing * (0.9 - 0.4 * facing) * ((0.5 + NoH) / max(NoH, 0.0001));

    float single = mix(smoothTerm, roughTerm, a);

    vec3 multi = 0.1159 * a2 * albedo;

    vec3 diff = (albedo * single + multi) * (1.0 / PI);
    return diff * NoL;
}

void main() {
    // Dry cells pull vertices to y=uEtaRef (vert shader), which leaves a horizontal sheet; discard
    // fragments where the sim has no water so land is not tinted by a fake sheet below hills.
    ivec2 sz = textureSize(uH, 0);
    vec2 uv = vec2(
        (vWorldPos.x + uHalfW) / (float(sz.x) * uDx),
        (vWorldPos.z + uHalfD) / (float(sz.y) * uDx));
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0))))
        discard;
    float hCell = texture(uH, uv).r;
    if (max(hCell, 0.0) < uWetDepthEps)
        discard;

    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    float W = uTime * uWaterAnimation;
    N = waterWavesNormal(vWorldPos, N, W);
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.001);

    float t = clamp(vDepth / 4.0, 0.0, 1.0);
    vec3 shallow = vec3(0.38, 0.82, 0.98);
    vec3 deep = vec3(0.08, 0.24, 0.44);
    vec3 base = mix(deep, shallow, t);
    vec3 sun = vec3(1.0, 0.97, 0.92);
    vec3 diffuse = hammonDiffuse(N, L, V, base, kRoughness) * sun;

    vec3 spec = vec3(0.0);
    if (nl > 0.0) {
        vec3 H = normalize(V + L);
        float D = distributionGGX(N, H, kRoughness);
        float G = geometrySmith(nv, nl, kRoughness);
        float vh = max(dot(V, H), 0.0);
        vec3 F = fresnelSchlick(vh, kF0);
        // Directional light: Lo = (D*G*F / (4*NdotV*NdotL)) * Li * NdotL  =>  D*G*F*Li / (4*NdotV)
        spec = (D * G * F) * sun / (4.0 * nv);
        spec *= 1.35;
    }

    // --- IBL (cubemap): diffuse ≈ high-mip irradiance; specular ≈ reflection + roughness→LOD ---
    float alpha = kRoughness * kRoughness;
    vec3 Renv = reflect(-V, N);
    float lodSpec = alpha * uEnvMaxMip;
    vec3 prefilteredSpec = textureLod(uEnvMap, Renv, lodSpec).rgb;
    float lodDiff = uEnvMaxMip * 0.88;
    vec3 irradiance = textureLod(uEnvMap, N, lodDiff).rgb;
    vec3 Fnv = fresnelSchlick(nv, kF0);
    vec3 iblDiffuse = irradiance * base * (vec3(1.0) - Fnv) * kIBLDiffuseMul;
    vec3 iblSpec = prefilteredSpec * Fnv * kIBLSpecMul;

    vec4 clipR = uReflViewProj * vec4(vWorldPos, 1.0);
    float rw = clipR.w;
    vec2 uvR = clipR.xy / rw * 0.5 + 0.5;
    float reflOk =
        step(1e-4, rw) * step(0.001, uvR.x) * step(uvR.x, 0.999) * step(0.001, uvR.y) * step(uvR.y, 0.999);
    vec3 planar = texture(uReflectionTex, uvR).rgb;
    vec3 iblSpecFinal = mix(iblSpec, planar * kIBLSpecMul, reflOk * kPlanarReflMix);

    // --- Screen-space refraction (Pass 1 color + depth), UV offset by water normal ---
    vec2 screenUV = vClipPos.xy / vClipPos.w * 0.5 + 0.5;
    vec2 refractUV = clamp(screenUV + N.xy * uDistortStrength, 0.001, 0.999);
    vec3 refractionSample = texture(uRefractionTex, refractUV).rgb;
    float sceneZ = texture(uRefractionDepth, refractUV).r;
    float waterZ = gl_FragCoord.z;
    float sceneLin = linearizeDepthBuf(sceneZ);
    float waterLin = linearizeDepthBuf(waterZ);
    float column = clamp(max(sceneLin - waterLin, 0.0), 0.0, 40.0);
    vec3 absorb = exp(-uAbsorptionColor * uAbsorptionDensity * column);
    vec3 refrTinted = refractionSample * absorb;
    refrTinted = mix(refrTinted, refractionSample, uTransparency);

    float R0 = pow((1.0 - uIOR) / (1.0 + uIOR), 2.0);
    float fresnel = R0 + (1.0 - R0) * pow(1.0 - nv, 5.0);
    fresnel = mix(fresnel, 1.0, uWaterReflectionsMix);

    vec3 reflectionSide = iblDiffuse + iblSpecFinal + spec;
    vec3 refractionSide = refrTinted + diffuse * 0.42;
    vec3 rgb = mix(refractionSide, reflectionSide, fresnel);
    rgb += uEmissionColor * uEmissionStrength;
    FragColor = vec4(rgb, uAlpha);
}