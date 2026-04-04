#version 330 core
in vec3 vWorldPos;
in float vDepth;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform float uAlpha;
out vec4 FragColor;

const float PI = 3.14159265;
// Lower = smoother water, tighter sun glint.
const float kRoughness = 0.045;
// Air–water interface, normal-incidence reflectance (~IOR 1.33).
const vec3 kF0 = vec3(0.02);
// F0 is tiny: without a scale the GGX term is hard to see next to strong diffuse + blending.
const float kSpecularArtisticScale = 7.0;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float nh = max(dot(N, H), 0.0);
    float nh2 = nh * nh;
    float denom = nh2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

float geometrySchlickGGX(float nx, float k) {
    return nx / (nx * (1.0 - k) + k);
}

float geometrySmith(float nv, float nl, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return geometrySchlickGGX(nv, k) * geometrySchlickGGX(nl, k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    return F0 + (1.0 - F0) * (t * t * t * t * t);
}

void main() {
    vec3 nx = dFdx(vWorldPos);
    vec3 ny = dFdy(vWorldPos);
    vec3 N = normalize(cross(nx, ny));
    vec3 V = normalize(uCameraPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    float nl = max(dot(N, L), 0.0);
    float nv = max(dot(N, V), 0.001);

    float t = clamp(vDepth / 4.0, 0.0, 1.0);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.14, 0.32);
    vec3 base = mix(deep, shallow, t);
    vec3 diffuse = base * (0.22 + 0.78 * nl);

    vec3 spec = vec3(0.0);
    if (nl > 0.0) {
        vec3 H = normalize(V + L);
        float D = distributionGGX(N, H, kRoughness);
        float G = geometrySmith(nv, nl, kRoughness);
        float vh = max(dot(V, H), 0.0);
        vec3 F = fresnelSchlick(vh, kF0);
        vec3 sun = vec3(1.0, 0.97, 0.92);
        // Directional light: Lo = (D*G*F / (4*NdotV*NdotL)) * Li * NdotL  =>  D*G*F*Li / (4*NdotV)
        spec = (D * G * F) * sun / (4.0 * nv) * kSpecularArtisticScale;
    }

    vec3 rgb = diffuse + spec;
    FragColor = vec4(rgb, uAlpha);
}