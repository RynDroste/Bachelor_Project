#version 410 core
in vec3 vNrm;
uniform vec3 uLightDir;
uniform vec3 uBaseColor;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNrm);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = uBaseColor * (0.25 + 0.75 * ndl);
    FragColor = vec4(rgb, 1.0);
}
