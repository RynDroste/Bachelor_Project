#version 330 core
in vec3 vWorldPos;
in vec2 vIJ;
uniform sampler2D uB;
uniform float uDx;
uniform vec3 uLightDir;
uniform vec3 uCamPos;
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
    float b = texelFetch(uB, ivec2(i, j), 0).r;
    // Tint vs B (generator: sea ~[0,3.9], beach ~[4.1,6], land ~[6,8]).
    vec3 colDeep = vec3(0.12, 0.14, 0.18);
    vec3 colSand = vec3(0.72, 0.62, 0.48);
    vec3 colRock = vec3(0.38, 0.36, 0.34);
    float tSand = smoothstep(3.2, 4.25, b);
    float tRock = smoothstep(5.9, 6.5, b);
    vec3 albedo = mix(colDeep, colSand, tSand);
    albedo = mix(albedo, colRock, tRock);
    vec3 V = normalize(uCamPos - vWorldPos);
    float spec = pow(max(dot(N, normalize(L + V)), 0.0), 32.0) * 0.12;
    vec3 rgb = albedo * (0.22 + 0.78 * ndl) + vec3(spec);
    FragColor = vec4(rgb, 1.0);
}
