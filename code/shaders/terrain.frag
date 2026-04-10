#version 410 core
in vec3 vWorldPos;
in vec2 vIJ;
in vec2 vUv;
uniform sampler2D uB;
uniform sampler2D uAlbedo;
uniform sampler2D uNormalMap;
uniform sampler2D uAO;
uniform sampler2D uRoughness;
uniform float uDx;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform sampler2D uH;
uniform sampler2D uCausticTex;
uniform float uCausticTime;
uniform vec2 uCausticWaveOffset;
out vec4 FragColor;

void main() {
    ivec2 sz = textureSize(uB, 0);
    int NX = sz.x;
    int NY = sz.y;
    int i = int(floor(vIJ.x + 0.0001));
    int j = int(floor(vIJ.y + 0.0001));
    i = clamp(i, 0, NX - 1);
    j = clamp(j, 0, NY - 1);
    int im = max(i - 1, 0);
    int ip = min(i + 1, NX - 1);
    int jm = max(j - 1, 0);
    int jp = min(j + 1, NY - 1);
    float denomX = float(ip - im) * uDx;
    float denomZ = float(jp - jm) * uDx;
    float dBdx = denomX > 1e-6
        ? (texelFetch(uB, ivec2(ip, j), 0).r - texelFetch(uB, ivec2(im, j), 0).r) / denomX
        : 0.0;
    float dBdz = denomZ > 1e-6
        ? (texelFetch(uB, ivec2(i, jp), 0).r - texelFetch(uB, ivec2(i, jm), 0).r) / denomZ
        : 0.0;
    vec3 Ng = normalize(vec3(-dBdx, 1.0, -dBdz));
    vec3 up = abs(Ng.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, Ng));
    vec3 B = cross(Ng, T);
    mat3 TBN = mat3(T, B, Ng);
    vec3 tN = texture(uNormalMap, vUv).rgb * 2.0 - 1.0;
    vec3 N = normalize(TBN * normalize(tN));

    vec3 albedo = texture(uAlbedo, vUv).rgb;
    float ao = texture(uAO, vUv).r;
    float rough = texture(uRoughness, vUv).r;

    vec3 L = normalize(-uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);
    float ndh = max(dot(N, H), 0.0);
    float shininess = mix(96.0, 4.0, rough);
    float specAmt = (1.0 - rough * 0.92) * 0.22;
    float spec = pow(ndh, shininess) * specAmt;

    const float kAmb = 0.32;
    const float kDir = 0.68;
    float diffuseLight = kAmb * ao + kDir * ndl;
    vec3 rgb = albedo * diffuseLight + vec3(spec);

    float waterDepth = texelFetch(uH, ivec2(i, j), 0).r;
    if (waterDepth > 0.05) {
        const float kCausticScale = 0.010;
        vec2 lightShift = -uLightDir.xz * waterDepth * 0.18;
        vec2 xz = vWorldPos.xz + lightShift;
        vec2 cBase = xz * kCausticScale + uCausticWaveOffset;
        vec2 cScroll1 = vec2(uCausticTime * 0.025, uCausticTime * 0.018);
        vec2 cScroll2 = vec2(-uCausticTime * 0.018, uCausticTime * 0.022);
        vec2 cUV1 = cBase + cScroll1;
        vec2 cUV2 = cBase * 1.45 + cScroll2;

        vec2 chDir = uLightDir.xz;
        float chLen2 = dot(chDir, chDir);
        chDir = chLen2 > 1e-8 ? chDir * inversesqrt(chLen2) : vec2(1.0, 0.0);
        const float kCausticAber = 0.0018;
        vec2 ab = chDir * kCausticAber;

        float c1r = texture(uCausticTex, cUV1 + ab).r;
        float c1g = texture(uCausticTex, cUV1).r;
        float c1b = texture(uCausticTex, cUV1 - ab).r;
        float c2r = texture(uCausticTex, cUV2 + ab).r;
        float c2g = texture(uCausticTex, cUV2).r;
        float c2b = texture(uCausticTex, cUV2 - ab).r;
        vec3 raw = vec3(sqrt(c1r * c2r), sqrt(c1g * c2g), sqrt(c1b * c2b));
        raw = pow(max(raw, vec3(1e-5)), vec3(1.65));
        raw = clamp(0.5 + (raw - 0.5) * 1.35, 0.0, 1.0);
        const float kCausticGain     = 2.95;
        const float kCausticCoverage = 0.52;
        float depthAtten = clamp(waterDepth / 4.0, 0.0, 1.0);
        vec3 caustic = raw * kCausticGain * (1.0 - depthAtten) * kCausticCoverage;
        rgb += caustic * vec3(1.03, 1.0, 0.98);
    }

    FragColor = vec4(rgb, 1.0);
}
