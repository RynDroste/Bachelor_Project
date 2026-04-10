#version 410 core
in vec3 vWorldPos;
in float vDepth;
in float vGerstnerPeak;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform samplerCube uEnvMap;
uniform float uEnvMaxMip;
uniform float uWaveRough;
uniform sampler2D uReflectionTex;
uniform sampler2D uRefractionTex;
uniform sampler2D uSceneDepth;
uniform vec2 uClipNF;
uniform float uDepthAbsorb;
uniform vec4 uViewport;
uniform mat3 uViewRot;
uniform float uIOR;
uniform float uTransparency;
uniform vec3 uWaterColor;
uniform float uWaterReflections;
uniform float uRefractionMaxOffset;
uniform float uRefractionLinTol;
uniform mat4 uReflViewProj;
uniform float uAlpha;
uniform int uWaterBodyLitOnly;
uniform float uDx;
uniform float uHalfW;
uniform float uHalfD;
uniform sampler2D uH;
uniform sampler2D uB;
uniform float uWetDepthEps;
uniform float uTime;
uniform float uWaveScale;
uniform float uWaveStrength;
uniform float uWaterAnimation;
uniform vec3 uEmissionColor;
uniform float uEmissionStrength;
out vec4 FragColor;

const float PI = 3.14159265;
const float kRoughness = 0.045;
const vec3 kF0 = vec3(0.02);
const float kIBLDiffuseMul = 0.38;
const float kIBLSpecMul    = 0.55;
const float kPlanarReflMix = 0.82;
const float kViewBodyMin = 0.05;
const float kBodyLitMul    = 1.65;
const float kFresnelEdgeBoost = 0.26;
// Nonlinear depth: pow < 1 → stronger shallow contrast, deep saturates earlier (typ. 0.3–0.6).
const float kDepthCurveMax  = 4.0;
const float kDepthCurvePow  = 0.45;
const float kShallowAlphaK  = 0.30;
const float kDeepAlphaK     = 0.95;
const float kGlassBaseTint  = 0.55;
// Blend SWE depth-gradient into shading normals so boat wakes read in spec/diffuse (mesh stays unchanged).
const float kSweHNormalStr = 1.4;
const float kSweHNormalMix = 0.58;

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

float hash11(float n) {
    return fract(sin(n) * 43758.5453123);
}

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

float noise4D(vec4 x) {
    float wi = floor(x.w);
    float wf = fract(x.w);
    wf = wf * wf * (3.0 - 2.0 * wf);
    vec3 p = x.xyz;
    float a = noise3(p + wi * vec3(17.13, 31.37, 11.71));
    float b = noise3(p + (wi + 1.0) * vec3(17.13, 31.37, 11.71));
    return mix(a, b, wf);
}

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

vec4 waterWaveCoord(vec3 worldPos, float W) {
    float s = uWaveScale;
    return vec4(worldPos.x * s, worldPos.z * 2.0 * s, 0.0, W);
}

vec3 waterWavesNormal(vec3 worldPos, vec3 Ngeom, float W) {
    if (uWaveStrength <= 1e-6)
        return Ngeom;
    const float eps = 0.001;
    vec4 c0 = waterWaveCoord(worldPos, W);
    float h0 = fbm4Raw(c0);
    float hx = fbm4Raw(c0 + vec4(eps, 0.0, 0.0, 0.0));
    float hz = fbm4Raw(c0 + vec4(0.0, eps, 0.0, 0.0));
    float gx = (hx - h0) / eps;
    float gz = (hz - h0) / eps;

    vec3 Tn = vec3(1.0, 0.0, 0.0) - Ngeom * Ngeom.x;
    if (dot(Tn, Tn) < 1e-8)
        Tn = vec3(0.0, 0.0, 1.0) - Ngeom * Ngeom.z;
    Tn = normalize(Tn);
    vec3 Bn = normalize(cross(Ngeom, Tn));
    return normalize(Ngeom - uWaveStrength * (gx * Tn + gz * Bn));
}

float fresnelSchlickIOR(float cosTheta, float ior) {
    float F0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    return F0 + (1.0 - F0) * (t * t * t * t * t);
}

float linearizeDepth(float z, float near, float far) {
    float zNdc = z * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - zNdc * (far - near));
}

vec3 glassBSDF(vec3 N, vec3 V, vec2 screenUV, vec2 refractOff, vec3 planarReflect, float reflOk,
               float waterDeviceZ, vec3 base) {
    float near = uClipNF.x;
    float far = uClipNF.y;
    float linWater = linearizeDepth(waterDeviceZ, near, far);

    vec2 offset = clamp(refractOff * 0.05 * uTransparency, vec2(-uRefractionMaxOffset), vec2(uRefractionMaxOffset));
    vec2 bentUV = clamp(screenUV + offset, vec2(0.001), vec2(0.999));

    vec2 refractUV = screenUV;
    float sceneZBent = texture(uSceneDepth, bentUV).r;
    float linBent = linearizeDepth(sceneZBent, near, far);
    if (linBent > linWater + uRefractionLinTol)
        refractUV = bentUV;

    vec3 refract = texture(uRefractionTex, refractUV).rgb * uWaterColor;
    float sceneZ = texture(uSceneDepth, refractUV).r;
    if (sceneZ < 0.999) {
        float linS = linearizeDepth(sceneZ, near, far);
        float thick = clamp(max(0.0, linS - linWater), 0.0, 40.0);
        refract *= exp(-uDepthAbsorb * thick);
    }

    refract *= mix(vec3(1.0), base, kGlassBaseTint);

    vec3 reflectDir = reflect(-V, N);
    float alpha = kRoughness * kRoughness;
    float mip = min(uWaveRough * uEnvMaxMip + alpha * uEnvMaxMip, uEnvMaxMip);
    vec3 envReflect = textureLod(uEnvMap, reflectDir, mip).rgb;
    vec3 reflectCombined = mix(envReflect * kIBLSpecMul, planarReflect * kIBLSpecMul, reflOk * kPlanarReflMix);

    float nvSurf = max(dot(N, V), 0.0);
    float f = fresnelSchlickIOR(nvSurf, uIOR);
    float edgeBoost = kFresnelEdgeBoost * pow(1.0 - nvSurf, 3.0);
    float reflW = clamp(f * uWaterReflections + edgeBoost, 0.0, 1.0);
    return mix(refract, reflectCombined, reflW);
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
    ivec2 sz = textureSize(uH, 0);
    vec2 uv = vec2(
        (vWorldPos.x + uHalfW) / (float(sz.x) * uDx),
        (vWorldPos.z + uHalfD) / (float(sz.y) * uDx));
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0))))
        discard;
    float hCell = texture(uH, uv).r;
    if (max(hCell, 0.0) < uWetDepthEps)
        discard;

    float bTerrain = texture(uB, uv).r;
    float surfaceToBed = max(vWorldPos.y - bTerrain, 0.0);
    float depthNorm = clamp(surfaceToBed / kDepthCurveMax, 0.0, 1.0);
    float depthFactor = pow(depthNorm, kDepthCurvePow);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.08, 0.32);
    vec3 base = mix(shallow, deep, depthFactor);
    float waterAlpha = clamp(mix(kShallowAlphaK, kDeepAlphaK, depthFactor) * uAlpha, 0.0, 1.0);

    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    float W = uTime * uWaterAnimation;
    N = waterWavesNormal(vWorldPos, N, W);
    {
        float cellDu = 1.0 / float(sz.x);
        float cellDv = 1.0 / float(sz.y);
        vec2 uxp = clamp(uv + vec2(cellDu, 0.0), vec2(0.001), vec2(0.999));
        vec2 uxm = clamp(uv - vec2(cellDu, 0.0), vec2(0.001), vec2(0.999));
        vec2 uyp = clamp(uv + vec2(0.0, cellDv), vec2(0.001), vec2(0.999));
        vec2 uym = clamp(uv - vec2(0.0, cellDv), vec2(0.001), vec2(0.999));
        float dhdx = (texture(uH, uxp).r - texture(uH, uxm).r) / (2.0 * uDx);
        float dhdz = (texture(uH, uyp).r - texture(uH, uym).r) / (2.0 * uDx);
        vec3 Nswe = normalize(vec3(-dhdx * kSweHNormalStr, 1.0, -dhdz * kSweHNormalStr));
        N = normalize(mix(N, Nswe, kSweHNormalMix));
    }
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.001);

    // --- Subsurface scatter: water_color = lerp(deep_water_color, subsurface_water_color, scatter_weight) ---
    // Factor 1 (view angle):  top-down view = shorter light path through water = more subsurface
    float sss_view = nv;
    // Factor 2 (sun direction): forward scatter — sun behind wave crest, light punches through thin water
    float sss_sun  = 0.5 + 0.5 * pow(max(dot(-L, V), 0.0), 2.0);
    // Factor 3 (wave peak mask): SWE shallow depth OR Gerstner crest — both mean short light path
    float sss_peak = pow(max(1.0 - depthFactor, vGerstnerPeak), 0.5);
    float scatter_weight = clamp(sss_view * sss_sun * sss_peak, 0.0, 1.0);
    vec3 sss_color = mix(deep, vec3(0.18, 0.82, 0.60), scatter_weight);
    vec3 sss       = sss_color * scatter_weight * 0.55;

    vec3 sun = vec3(1.0, 0.97, 0.92);
    vec3 diffuse = hammonDiffuse(N, L, V, base, kRoughness) * sun;

    vec3 spec = vec3(0.0);
    if (nl > 0.0) {
        vec3 H = normalize(V + L);
        float D = distributionGGX(N, H, kRoughness);
        float G = geometrySmith(nv, nl, kRoughness);
        float vh = max(dot(V, H), 0.0);
        vec3 F = fresnelSchlick(vh, kF0);
        spec = (D * G * F) * sun / (4.0 * nv);
        spec *= 1.35;
        spec *= 1.0 + 2.2 * pow(1.0 - nv, 2.5);
    }

    float lodDiff = uEnvMaxMip * 0.88;
    vec3 irradiance = textureLod(uEnvMap, N, lodDiff).rgb;
    float F0ior = pow((1.0 - uIOR) / (1.0 + uIOR), 2.0);
    vec3 Fnv = fresnelSchlick(nv, vec3(F0ior));
    vec3 iblDiffuse = irradiance * base * (vec3(1.0) - Fnv) * kIBLDiffuseMul;

    float viewBody = 4.0 * nv * (1.0 - nv);
    viewBody = clamp(viewBody, 0.0, 1.0);
    vec3 bodyLit = (diffuse + iblDiffuse) * mix(kViewBodyMin, 1.0, viewBody);
    bodyLit *= kBodyLitMul;

    float shallowMask = 1.0 - clamp(vDepth / 5.0, 0.0, 1.0);
    float edgeMask = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    vec3 emission = uEmissionColor * uEmissionStrength * (0.6 + 0.4 * edgeMask) * shallowMask;

    if (uWaterBodyLitOnly != 0) {
        FragColor = vec4(bodyLit + spec + emission + sss, waterAlpha);
        return;
    }

    vec4 clipR = uReflViewProj * vec4(vWorldPos, 1.0);
    float rw = clipR.w;
    vec2 uvR = clipR.xy / rw * 0.5 + 0.5;
    float reflOk =
        step(1e-4, rw) * step(0.001, uvR.x) * step(uvR.x, 0.999) * step(0.001, uvR.y) * step(uvR.y, 0.999);
    vec3 planar = texture(uReflectionTex, uvR).rgb;

    vec2 screenUV = vec2(
        (gl_FragCoord.x - uViewport.x) / max(uViewport.z, 1.0),
        gl_FragCoord.y / max(uViewport.w, 1.0));
    vec3 Nh = vec3(N.x, 0.0, N.z);
    float nh2 = dot(Nh, Nh);
    vec2 refractOff = vec2(0.0);
    if (nh2 > 1e-10) {
        vec3 NhN = Nh * inversesqrt(nh2);
        refractOff = (uViewRot * NhN).xy;
    }

    vec3 glass = glassBSDF(N, V, screenUV, refractOff, planar, reflOk, gl_FragCoord.z, base);

    vec3 rgb = bodyLit + spec + glass + emission + sss;

    FragColor = vec4(rgb, waterAlpha);
}