#version 330 core
in vec3 vWorldPos;
in float vDepth;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform samplerCube uEnvMap;
uniform float uEnvMaxMip;
uniform sampler2D uReflectionTex;
uniform mat4 uReflViewProj;
uniform float uAlpha;
out vec4 FragColor;

const float PI = 3.14159265;
// Perceptual roughness [0,1]; shared by Hammon diffuse and GGX specular (α = roughness²).
const float kRoughness = 0.045;
// Air–water interface, normal-incidence reflectance (~IOR 1.33).
const vec3 kF0 = vec3(0.02);
// IBL: cubemap mips approximate prefilter; no BRDF LUT (split-sum) — strengths are artistic knobs.
const float kIBLDiffuseMul = 0.5;
const float kIBLSpecMul    = 0.55;
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
    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.001);

    float t = clamp(vDepth / 4.0, 0.0, 1.0);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.14, 0.32);
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

    vec3 rgb = diffuse + spec + iblDiffuse + iblSpecFinal;
    FragColor = vec4(rgb, uAlpha);
}