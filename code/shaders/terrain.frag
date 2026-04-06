#version 410 core
in vec3 vWorldPos;
in vec2 vIJ;
uniform sampler2D uB;
uniform sampler2D uAlbedo;
uniform float uDx;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
uniform float uAlbedoScale;
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
    vec3 N = normalize(vec3(-dBdx, 1.0, -dBdz));
    vec3 L = normalize(-uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec2 uv = vec2(vWorldPos.x, vWorldPos.z) / max(uAlbedoScale, 1e-4);
    vec3 albedo = texture(uAlbedo, uv).rgb;
    vec3 V = normalize(uCamPos - vWorldPos);
    float spec = pow(max(dot(N, normalize(L + V)), 0.0), 32.0) * 0.12;
    // 贴图里的“黄沙滩”是偏亮的中性色；乘上过低的 (ambient + diffuse) 会变成暗黄褐。
    // 略提高环境项 + 半球天空感，避免整片压成棕色。
    float skyAmt = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 amb = mix(vec3(0.28, 0.26, 0.24), vec3(0.42, 0.45, 0.48), skyAmt);
    vec3 rgb = albedo * (amb + vec3(0.55 * ndl)) + vec3(spec);
    FragColor = vec4(rgb, 1.0);
}
