#version 410 core
in vec3 vNrm;
in vec4 vLightSpacePos;
uniform vec3 uLightDir;
uniform vec3 uBaseColor;
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
    vec3 N = normalize(vNrm);
    vec3 L = normalize(-uLightDir);
    float ndl = max(dot(N, L), 0.0);
    float sh = shadowPCF(uShadowMap, vLightSpacePos, N, L);
    vec3 rgb = uBaseColor * (0.28 + 0.72 * ndl * sh);
    FragColor = vec4(rgb, 1.0);
}
