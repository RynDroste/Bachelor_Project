#version 430

#define ONE_OVER_4PI	0.0795774715459476

uniform samplerCube	envmap;
uniform sampler2D	gradients;
uniform sampler2D	refractionTex;
uniform sampler2D	sceneDepth;

in vec3			vdir;
in vec2			tex;
in vec3			vertex;
flat in int		lod;

uniform vec3	oceanColor;
uniform float	transparency;
uniform vec3	sunColor;
uniform vec3	sunDir;
uniform int		bEnvmap;
uniform float	exposure;
uniform bool	bAbsorbance;
uniform vec3	absorbanceColor;
uniform float	absorbanceCoeff;
uniform vec3	eyePos;
uniform bool	bShowPatch;
uniform bool	bUseScreenRefraction;
uniform vec2	clipNF;
uniform float	depthAbsorb;
uniform vec4	viewport;        // x,y,w,h
uniform mat3	viewRot;
uniform float	waterIOR;
uniform vec3	waterColor;
uniform float	waterReflections;
uniform float	refractionMaxOffset;
uniform float	refractionLinTol;

out vec4 FragColor;

const float kDepthCurvePow = 0.45;
const float kShallowAlphaK = 0.30;
const float kDeepAlphaK    = 0.95;
const float kPlanarReflMix = 0.82;
const float kFresnelEdgeBoost = 0.26;

float fresnelSchlickIOR(float cosTheta, float ior) {
    float f0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
    float t = clamp(1.0 - cosTheta, 0.0, 1.0);
    return f0 + (1.0 - f0) * (t * t * t * t * t);
}

float linearizeDepth(float z, float near, float far) {
    float zNdc = z * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - zNdc * (far - near));
}

void main()
{
	vec4 grad = texture(gradients, tex);
	vec3 N = normalize(grad.xzy);
	vec3 V = normalize(vdir);
	vec3 R = reflect(-V, N);

    float nv = max(dot(N, V), 0.001);
    float underwater = step(eyePos.y, vertex.y);
    float wavePeak = clamp(1.0 - grad.w, 0.0, 1.0);
    float depthFactor = pow(clamp(grad.w, 0.0, 1.0), kDepthCurvePow);
    vec3 shallow = vec3(0.28, 0.75, 0.95);
    vec3 deep = vec3(0.02, 0.08, 0.32);
    vec3 base = mix(shallow, deep, depthFactor);
    float waterAlpha = clamp(mix(kShallowAlphaK, kDeepAlphaK, depthFactor) * transparency, 0.0, 1.0);

    float sss_view = nv;
    float sss_sun = 0.5 + 0.5 * pow(max(dot(-sunDir, V), 0.0), 2.0);
    float sss_peak = pow(max(1.0 - depthFactor, wavePeak), 0.5);
    float scatter_weight = clamp(sss_view * sss_sun * sss_peak, 0.0, 1.0);
    vec3 sss_color = mix(deep, vec3(0.18, 0.82, 0.60), scatter_weight);
    vec3 sss = sss_color * scatter_weight * 0.55;

	vec3 reflection = 0.5 * texture(envmap, R).rgb;
	if (bEnvmap == 0)
	{
		reflection = mix(base, sunColor, R.y * 0.5 + 0.5) * 2;
		float sunInfluence = max(dot(R, sunDir), 0.0);
		reflection += sunColor * pow(sunInfluence, 32.0);
	}

	float turbulence = max(grad.w + 0.8, 0.0);
	float color_mod = 1.0 + 3.0 * smoothstep(1.2, 1.8, turbulence);
    vec3 reflectionMod = reflection * color_mod;

    vec3 refraction = mix(base, reflectionMod, 0.12);
    if (bUseScreenRefraction)
    {
        vec2 screenUV = vec2(
            (gl_FragCoord.x - viewport.x) / max(viewport.z, 1.0),
            (gl_FragCoord.y - viewport.y) / max(viewport.w, 1.0)
        );
        vec3 Nh = vec3(N.x, 0.0, N.z);
        float nh2 = dot(Nh, Nh);
        vec2 refractOff = vec2(0.0);
        if (nh2 > 1e-10) {
            vec3 NhN = Nh * inversesqrt(nh2);
            refractOff = (viewRot * NhN).xy;
        }

        float linWater = linearizeDepth(gl_FragCoord.z, clipNF.x, clipNF.y);
        float refractStrength = mix(0.05, 0.18, underwater) * transparency;
        vec2 offset = clamp(refractOff * refractStrength, vec2(-refractionMaxOffset), vec2(refractionMaxOffset));
        vec2 bentUV = clamp(screenUV + offset, vec2(0.001), vec2(0.999));
        vec2 refractUV = screenUV;

        float sceneZBent = texture(sceneDepth, bentUV).r;
        float linBent = linearizeDepth(sceneZBent, clipNF.x, clipNF.y);
        if (underwater > 0.5 || linBent > linWater + refractionLinTol)
            refractUV = bentUV;

        refraction = texture(refractionTex, refractUV).rgb * waterColor;
        float sceneZ = texture(sceneDepth, refractUV).r;
        if (sceneZ < 0.999) {
            float linScene = linearizeDepth(sceneZ, clipNF.x, clipNF.y);
            float thick = clamp(max(0.0, linScene - linWater), 0.0, 40.0);
            refraction *= exp(-depthAbsorb * thick);
        }
        refraction *= mix(vec3(1.0), base, 0.55);
    }

    float f = fresnelSchlickIOR(nv, waterIOR);
    float edgeBoost = kFresnelEdgeBoost * pow(1.0 - nv, 3.0);
    float reflectW = clamp(f * waterReflections + edgeBoost, 0.0, 1.0);
    // Keep underwater mostly refractive while preserving a little reflection.
    reflectW *= mix(1.0, 0.02, underwater);
    vec3 glass = mix(refraction, reflectionMod, reflectW * kPlanarReflMix);

	float spec = 0.0;
	if (sunDir.y > 0.0)
	{	
		const float rho = 0.3;
		const float ax = 0.2;
		const float ay = 0.1;

		vec3 h = sunDir + V;
		vec3 x = cross(sunDir, N);
		vec3 y = cross(x, N);

		float mult = (ONE_OVER_4PI * rho / (ax * ay * sqrt(max(1e-5, dot(sunDir, N) * dot(V, N)))));
		float hdotx = dot(h, x) / ax;
		float hdoty = dot(h, y) / ay;
		float hdotn = dot(h, N);

		spec = mult * exp(-((hdotx * hdotx) + (hdoty * hdoty)) / (hdotn * hdotn));
	}

	FragColor = vec4(base + glass + sunColor * spec + sss, waterAlpha);

	if (bAbsorbance)
	{
		float distanceToCamera = length(eyePos - vertex);
		float absorbanceFactor = exp(-absorbanceCoeff * distanceToCamera);
		absorbanceFactor = clamp(absorbanceFactor, 0.0, 1.0);
		vec3 finalColor = mix(absorbanceColor * exposure, FragColor.rgb, absorbanceFactor);
		FragColor = vec4(finalColor, FragColor.a);
	}

	if (bShowPatch)
	{
		float borderWidth = 0.05;
		float edge = max( smoothstep(1.0 - borderWidth, 1.0, tex.x), smoothstep(1.0 - borderWidth, 1.0, tex.y) );
		vec3 borderColor = vec3(1.0, 0.0, 0.0);
		switch(int(floor(lod)))
		{
		case 0: borderColor = vec3(1.0, 0.0, 0.0); break;
		case 1: borderColor = vec3(0.0, 1.0, 0.0); break;
		case 2: borderColor = vec3(0.0, 0.0, 1.0); break;
		case 3: borderColor = vec3(1.0, 0.0, 1.0); break;
		case 4: borderColor = vec3(0.0, 1.0, 1.0); break;
		}
		FragColor.rgb = mix(FragColor.rgb, borderColor, edge);
	}

	if (underwater > 0.5) 
	{
		vec3 underwaterColor = vec3(0.0, 0.18, 0.22);
		float depth = clamp((vertex.y - eyePos.y) / 35.0, 0.0, 1.0);
		FragColor.rgb = mix(FragColor.rgb, underwaterColor, depth);
        // Underwater: avoid blending-through to the original background image.
        FragColor.a = 1.0;
	}

    FragColor.rgb = vec3(1.0) - exp(-FragColor.rgb * exposure);
}
