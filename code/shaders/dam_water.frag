#version 410 core
in vec3 vWorldPos;
in float vDepth;
in float vWetFrac;
uniform vec3 uLightDir;
uniform float uAlpha;
uniform float uWetDepthEps;
out vec4 FragColor;
void main() {
    // Soft edge: avoid a hard horizontal "void" cut when h or wet corner count is marginal.
    float dFade = smoothstep(0.0, uWetDepthEps * 2.5, max(vDepth, 0.0));
    float wFade = smoothstep(0.10, 0.30, vWetFrac);
    float a = uAlpha * dFade * wFade;
    if (a < 0.012)
        discard;
    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    float t = clamp(vDepth / 5.0, 0.0, 1.0);
    vec3 shallow = vec3(0.32, 0.78, 0.96);
    vec3 deep = vec3(0.03, 0.14, 0.33);
    vec3 base = mix(deep, shallow, t);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = base * (0.2 + 0.8 * ndl);
    FragColor = vec4(rgb, a);
}
