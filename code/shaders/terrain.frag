#version 410 core
in vec3 vWorldPos;
in vec2 vIJ;
in vec2 vUv;
in vec4 vLightSpacePos;
uniform sampler2D uB;
uniform sampler2D uAlbedo;
uniform sampler2D uNormalMap;
uniform sampler2D uAO;
uniform sampler2D uRoughness;
uniform float uDx;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform sampler2D uShadowMap;
out vec4 FragColor;

float shadowPCF(sampler2D map, vec4 fragLS, vec3 N, vec3 Lsurf) {
    vec3 proj = fragLS.xyz / fragLS.w;
    proj = proj * 0.5 + 0.5;
    if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0 || proj.z > 1.0)
        return 1.0;
    float bias = max(0.0012 * (1.0 - dot(N, Lsurf)), 0.00025);
    float cur = proj.z - bias;
    float vis = 0.0;
    vec2 ts = 1.0 / vec2(textureSize(map, 0));
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float d = texture(map, proj.xy + vec2(float(i), float(j)) * ts).r;
            vis += cur > d ? 0.0 : 1.0;
        }
    }
    return vis / 9.0;
}

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
    float sh = shadowPCF(uShadowMap, vLightSpacePos, N, L);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);
    float ndh = max(dot(N, H), 0.0);
    float shininess = mix(96.0, 4.0, rough);
    float specAmt = (1.0 - rough * 0.92) * 0.22;
    float spec = pow(ndh, shininess) * specAmt * sh;

    // AO only modulates indirect/ambient; direct sun uses N·L alone (avoids double darkening vs baked albedo).
    const float kAmb = 0.32;
    const float kDir = 0.68;
    float diffuseLight = kAmb * ao + kDir * ndl * sh;
    vec3 rgb = albedo * diffuseLight + vec3(spec);
    FragColor = vec4(rgb, 1.0);
}
