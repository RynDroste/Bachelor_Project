#version 330 core
in vec3 vWorldPos;
in float vDepth;
uniform vec3 uLightDir;
uniform float uAlpha;
out vec4 FragColor;
void main() {
    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    float t = clamp(vDepth / 4.0, 0.0, 1.0);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.14, 0.32);
    vec3 base = mix(deep, shallow, t);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = base * (0.22 + 0.78 * ndl);
    FragColor = vec4(rgb, uAlpha);
}
