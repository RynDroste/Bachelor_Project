#version 410 core

in vec3 vDir;
out vec4 fragColor;

uniform samplerCube uSky;

void main() {
    vec3 c = texture(uSky, normalize(vDir)).rgb;
    fragColor = vec4(c, 1.0);
}
