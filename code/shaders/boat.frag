#version 410 core
in vec3 vNrm;
in vec3 vWorldPos;
uniform vec3 uLightDir;
uniform vec3 uBaseColor;
// Underwater fog (ignored when uUnderwater == 0).
uniform int   uUnderwater;
uniform float uFogSigma;
uniform vec3  uFogColor;
uniform vec3  uCamPos;
out vec4 FragColor;
void main() {
    vec3 N = normalize(vNrm);
    vec3 L = normalize(uLightDir);
    float ndl = max(dot(N, L), 0.0);
    vec3 rgb = uBaseColor * (0.28 + 0.72 * ndl);
    if (uUnderwater != 0) {
        float dist  = length(uCamPos - vWorldPos);
        float atten = exp(-max(uFogSigma, 0.0) * dist);
        rgb = mix(uFogColor, rgb, atten);
    }
    FragColor = vec4(rgb, 1.0);
}
