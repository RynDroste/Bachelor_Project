#pragma once

#include <vector>

#include <glad/glad.h>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

struct Grid;

// FFT 海面（改编自 SimShip Ocean）：不含船舶 / Kelvin / 尾迹 / 泡沫。
struct SimshipOceanLighting {
    glm::vec3 sunDir{0.35f, 0.85f, 0.4f};
    glm::vec3 sunColor{1.0f, 0.98f, 0.92f};
    float     exposure{1.0f};
    float     waterLevel{0.0f};
    bool      useEnvCubemap{true};
    bool      useScreenRefraction{false};
    GLuint    refractionTex{0};
    GLuint    sceneDepthTex{0};
    glm::vec4 viewport{0.f, 0.f, 1.f, 1.f};  // x,y,w,h
    glm::mat3 viewRot{1.0f};
    float     clipNear{0.1f};
    float     clipFar{500.0f};
    float     depthAbsorb{0.008f};
    float     refractionMaxOffset{0.06f};
    float     refractionLinTol{0.12f};
    float     ior{1.33f};
    glm::vec3 waterColor{0.93f, 0.97f, 1.0f};
    float     waterReflections{0.86f};
    bool      absorbance{false};
    glm::vec3 absorbanceColor{0.71f, 0.84f, 1.0f};
    float     absorbanceCoeff{0.0015f};
};

class SimshipOcean {
public:
    SimshipOcean() = default;
    ~SimshipOcean();

    SimshipOcean(const SimshipOcean&)            = delete;
    SimshipOcean& operator=(const SimshipOcean&) = delete;

    // envCubemap: 天空盒立方体贴图（可与 skybox 共用）
    bool init(GLuint envCubemap);

    void shutdown();

    bool ok() const { return initialized_; }

    void update(float timeSeconds);
    void uploadCoupledHeightfield(const Grid& g, float etaRef);

    void render(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& eye,
                const SimshipOceanLighting& light);

    void setDebugShowPatch(bool show) { debugShowPatch_ = show; }

private:
    bool buildShaders_();
    void initSpectrumTextures_();
    void createBaseMesh_();
    void createLodMeshes_();
    void fourier2D_(GLuint spectrumTex);

    bool   initialized_ = false;
    GLuint envCubemap_  = 0;

    static constexpr int   FFT_SIZE     = 512;
    static constexpr int   FFT_SIZE_1   = FFT_SIZE + 1;
    static constexpr int   MESH_SIZE    = 256;
    static constexpr int   MESH_SIZE_1  = MESH_SIZE + 1;
    static constexpr int   PATCH_SIZE   = 100;
    static constexpr int   NbPatches    = 300;
    static constexpr float LengthWave   = 60.f;
    static constexpr float mGravity     = 9.81f;

    glm::vec2 wind_{0.f, 1.f};
    float     amplitude_{1.f};
    float     lambdaChop_{-0.75f};
    glm::vec3 oceanColor_{1.f / 255.f, 53.f / 255.f, 75.f / 255.f};
    float     transparency_{0.76f};
    float     wakeAmplitude_{2.0f};
    float     wakeChop_{6.0f};
    bool      wakeEnabled_{true};
    bool      debugShowPatch_{false};

    GLuint mTexInitialSpectrum_ = 0;
    GLuint mTexFrequencies_     = 0;
    GLuint mTexUpdatedSpectra_[2]{0, 0};
    GLuint mTexTempData_        = 0;
    GLuint mTexDisplacements_   = 0;
    GLuint mTexGradients_       = 0;
    GLuint mTexWake_            = 0;

    GLuint mVao_        = 0;
    GLuint mVbo_        = 0;
    GLuint mIbo_        = 0;
    int    mIndexCount_ = 0;

    std::vector<GLuint> lodVaos_;
    std::vector<GLuint> lodIndexCounts_;

    GLuint csSpectrum_      = 0;
    GLuint csFft_           = 0;
    GLuint csDisplacement_  = 0;
    GLuint csGradients_     = 0;
    GLuint progOcean_       = 0;

    GLint locMatViewProj_     = -1;
    GLint locEyePos_          = -1;
    GLint locOceanColor_      = -1;
    GLint locTransparency_    = -1;
    GLint locSunColor_        = -1;
    GLint locSunDir_          = -1;
    GLint locWaterLevel_      = -1;
    GLint locBUseScreenRefraction_ = -1;
    GLint locRefractionTex_        = -1;
    GLint locSceneDepth_           = -1;
    GLint locClipNF_               = -1;
    GLint locDepthAbsorb_          = -1;
    GLint locViewport_             = -1;
    GLint locViewRot_              = -1;
    GLint locWaterIOR_             = -1;
    GLint locWaterColor_           = -1;
    GLint locWaterReflections_     = -1;
    GLint locRefractionMaxOffset_  = -1;
    GLint locRefractionLinTol_     = -1;
    GLint locBEnvmap_        = -1;
    GLint locExposure_       = -1;
    GLint locBAbsorbance_    = -1;
    GLint locAbsorbanceColor_ = -1;
    GLint locAbsorbanceCoeff_ = -1;
    GLint locBShowPatch_       = -1;
};
