#version 330 core
in vec3 TexCoords;
uniform samplerCube uSkyMap;
out vec4 FragColor;
void main() {
    FragColor = texture(uSkyMap, TexCoords);
}
