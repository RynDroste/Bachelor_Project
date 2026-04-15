#include "render/big_surface.h"

#include "render/shader_file.h"
#include "solver_pipeline/shallow_water_solver.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

namespace {

struct GridVertex {
    glm::vec3 position{};
    glm::vec2 texCoord{};
};

struct InstanceData {
    glm::mat4 modelMatrix{1.f};
    int       lod{};
    int       pad_[3]{};  // stride 80：保证下一实例的 mat4 按 16 字节对齐
};

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "simship_ocean shader compile failed:\n%s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkProgramVsFs(GLuint vs, GLuint fs) {
    if (!vs || !fs)
        return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::fprintf(stderr, "simship_ocean program link failed:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

GLuint linkCompute(GLuint cs) {
    if (!cs)
        return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, cs);
    glLinkProgram(p);
    glDeleteShader(cs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        std::fprintf(stderr, "simship_ocean compute link failed:\n%s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

uint32_t highestSetBit(uint32_t x) {
    uint32_t ret = 0;
    while (x >>= 1u)
        ++ret;
    return ret;
}

float phillips(glm::vec2 k, glm::vec2 wind, float gravity) {
    float kLen = glm::length(k);
    if (kLen < 1e-6f)
        return 0.f;
    float k2  = kLen * kLen;
    float k4  = k2 * k2;
    float kdw = glm::dot(glm::normalize(k), glm::normalize(wind * 0.7f));
    if (kdw < 0.f)
        return 0.f;
    kdw = std::pow(std::cos(1.f * std::acos(kdw)), 3.f);
    float kdw2 = kdw * kdw;
    glm::vec2 w07 = wind * 0.7f;
    float     L   = glm::dot(w07, w07) / gravity;
    float L2   = L * L;
    float damp = 0.0001f;
    float l2   = L2 * damp * damp;
    float S    = std::exp(-1.f / (k2 * L2)) / k4 * kdw2 * std::exp(-k2 * l2);
    return S * 0.0000375f;
}

}  // namespace

SimshipOcean::~SimshipOcean() {
    shutdown();
}

bool SimshipOcean::init(GLuint envCubemap) {
    shutdown();
    envCubemap_ = envCubemap;

    glGenTextures(1, &mTexInitialSpectrum_);
    glGenTextures(1, &mTexFrequencies_);
    glGenTextures(2, mTexUpdatedSpectra_);
    glGenTextures(1, &mTexTempData_);
    glGenTextures(1, &mTexDisplacements_);
    glGenTextures(1, &mTexGradients_);
    glGenTextures(1, &mTexWake_);

    glBindTexture(GL_TEXTURE_2D, mTexInitialSpectrum_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, FFT_SIZE_1, FFT_SIZE_1);

    glBindTexture(GL_TEXTURE_2D, mTexFrequencies_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, FFT_SIZE_1, FFT_SIZE_1);

    for (int i = 0; i < 2; ++i) {
        glBindTexture(GL_TEXTURE_2D, mTexUpdatedSpectra_[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, FFT_SIZE, FFT_SIZE);
    }

    glBindTexture(GL_TEXTURE_2D, mTexTempData_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG32F, FFT_SIZE, FFT_SIZE);

    glBindTexture(GL_TEXTURE_2D, mTexDisplacements_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, FFT_SIZE, FFT_SIZE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, mTexGradients_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, FFT_SIZE, FFT_SIZE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glBindTexture(GL_TEXTURE_2D, mTexWake_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, FFT_SIZE, FFT_SIZE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    std::vector<float> wakeZeros(static_cast<size_t>(FFT_SIZE) * static_cast<size_t>(FFT_SIZE), 0.0f);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, FFT_SIZE, FFT_SIZE, GL_RED, GL_FLOAT, wakeZeros.data());

    glBindTexture(GL_TEXTURE_2D, 0);

    initSpectrumTextures_();
    createBaseMesh_();
    createLodMeshes_();

    if (!buildShaders_())
        return false;

    locMatViewProj_     = glGetUniformLocation(progOcean_, "matViewProj");
    locEyePos_          = glGetUniformLocation(progOcean_, "eyePos");
    locOceanColor_      = glGetUniformLocation(progOcean_, "oceanColor");
    locTransparency_    = glGetUniformLocation(progOcean_, "transparency");
    locSunColor_        = glGetUniformLocation(progOcean_, "sunColor");
    locSunDir_          = glGetUniformLocation(progOcean_, "sunDir");
    locWaterLevel_      = glGetUniformLocation(progOcean_, "waterLevel");
    locBUseScreenRefraction_ = glGetUniformLocation(progOcean_, "bUseScreenRefraction");
    locRefractionTex_        = glGetUniformLocation(progOcean_, "refractionTex");
    locSceneDepth_           = glGetUniformLocation(progOcean_, "sceneDepth");
    locClipNF_               = glGetUniformLocation(progOcean_, "clipNF");
    locDepthAbsorb_          = glGetUniformLocation(progOcean_, "depthAbsorb");
    locViewport_             = glGetUniformLocation(progOcean_, "viewport");
    locViewRot_              = glGetUniformLocation(progOcean_, "viewRot");
    locWaterIOR_             = glGetUniformLocation(progOcean_, "waterIOR");
    locWaterColor_           = glGetUniformLocation(progOcean_, "waterColor");
    locWaterReflections_     = glGetUniformLocation(progOcean_, "waterReflections");
    locRefractionMaxOffset_  = glGetUniformLocation(progOcean_, "refractionMaxOffset");
    locRefractionLinTol_     = glGetUniformLocation(progOcean_, "refractionLinTol");
    locBEnvmap_         = glGetUniformLocation(progOcean_, "bEnvmap");
    locExposure_        = glGetUniformLocation(progOcean_, "exposure");
    locBAbsorbance_     = glGetUniformLocation(progOcean_, "bAbsorbance");
    locAbsorbanceColor_ = glGetUniformLocation(progOcean_, "absorbanceColor");
    locAbsorbanceCoeff_ = glGetUniformLocation(progOcean_, "absorbanceCoeff");
    locBShowPatch_      = glGetUniformLocation(progOcean_, "bShowPatch");

    glUseProgram(progOcean_);
    glUniform1i(glGetUniformLocation(progOcean_, "displacement"), 0);
    glUniform1i(glGetUniformLocation(progOcean_, "envmap"), 1);
    glUniform1i(glGetUniformLocation(progOcean_, "gradients"), 2);
    glUniform1i(glGetUniformLocation(progOcean_, "refractionTex"), 3);
    glUniform1i(glGetUniformLocation(progOcean_, "sceneDepth"), 4);
    glUseProgram(0);

    initialized_ = true;
    return true;
}

void SimshipOcean::shutdown() {
    if (mTexInitialSpectrum_)
        glDeleteTextures(1, &mTexInitialSpectrum_);
    if (mTexFrequencies_)
        glDeleteTextures(1, &mTexFrequencies_);
    if (mTexUpdatedSpectra_[0])
        glDeleteTextures(2, mTexUpdatedSpectra_);
    if (mTexTempData_)
        glDeleteTextures(1, &mTexTempData_);
    if (mTexDisplacements_)
        glDeleteTextures(1, &mTexDisplacements_);
    if (mTexGradients_)
        glDeleteTextures(1, &mTexGradients_);
    if (mTexWake_)
        glDeleteTextures(1, &mTexWake_);
    mTexInitialSpectrum_ = mTexFrequencies_ = mTexTempData_ = mTexDisplacements_ = mTexGradients_ = mTexWake_ = 0;
    mTexUpdatedSpectra_[0] = mTexUpdatedSpectra_[1] = 0;

    if (mVao_)
        glDeleteVertexArrays(1, &mVao_);
    if (mVbo_)
        glDeleteBuffers(1, &mVbo_);
    if (mIbo_)
        glDeleteBuffers(1, &mIbo_);
    mVao_ = mVbo_ = mIbo_ = 0;

    for (GLuint v : lodVaos_)
        if (v)
            glDeleteVertexArrays(1, &v);
    lodVaos_.clear();
    lodIndexCounts_.clear();

    if (csSpectrum_)
        glDeleteProgram(csSpectrum_);
    if (csFft_)
        glDeleteProgram(csFft_);
    if (csDisplacement_)
        glDeleteProgram(csDisplacement_);
    if (csGradients_)
        glDeleteProgram(csGradients_);
    if (progOcean_)
        glDeleteProgram(progOcean_);
    csSpectrum_ = csFft_ = csDisplacement_ = csGradients_ = progOcean_ = 0;

    initialized_ = false;
}

void SimshipOcean::initSpectrumTextures_() {
    std::mt19937                          gen(static_cast<unsigned>(std::time(nullptr)));
    std::normal_distribution<double>     gaussian(0.0, 1.0);
    std::vector<std::complex<float>> h0(static_cast<size_t>(FFT_SIZE_1) * static_cast<size_t>(FFT_SIZE_1));
    std::vector<float>                   w(static_cast<size_t>(FFT_SIZE_1) * static_cast<size_t>(FFT_SIZE_1));

    for (int m = 0; m <= FFT_SIZE; ++m) {
        for (int n = 0; n <= FFT_SIZE; ++n) {
            glm::vec2 k;
            constexpr float pi = 3.14159265358979323846f;
            k.x = 2.f * pi * (n - FFT_SIZE / 2) / LengthWave;
            k.y = 2.f * pi * (m - FFT_SIZE / 2) / LengthWave;
            float sqrtS = std::sqrt(phillips(k, wind_, mGravity)) * amplitude_;
            int   idx   = m * FFT_SIZE_1 + n;
            h0[static_cast<size_t>(idx)] =
                std::complex<float>(static_cast<float>(gaussian(gen) * sqrtS), static_cast<float>(gaussian(gen) * sqrtS));
            w[static_cast<size_t>(idx)] = std::sqrt(mGravity * glm::length(k));
        }
    }

    glBindTexture(GL_TEXTURE_2D, mTexFrequencies_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, FFT_SIZE_1, FFT_SIZE_1, GL_RED, GL_FLOAT, w.data());

    std::vector<float> h0rg(h0.size() * 2);
    for (size_t i = 0; i < h0.size(); ++i) {
        h0rg[i * 2]     = h0[i].real();
        h0rg[i * 2 + 1] = h0[i].imag();
    }
    glBindTexture(GL_TEXTURE_2D, mTexInitialSpectrum_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, FFT_SIZE_1, FFT_SIZE_1, GL_RG, GL_FLOAT, h0rg.data());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void SimshipOcean::createBaseMesh_() {
    std::vector<GridVertex> vdata(static_cast<size_t>(MESH_SIZE_1) * static_cast<size_t>(MESH_SIZE_1));
    for (int z = 0; z <= MESH_SIZE; ++z) {
        for (int x = 0; x <= MESH_SIZE; ++x) {
            int idx             = z * MESH_SIZE_1 + x;
            vdata[idx].position = glm::vec3((x - MESH_SIZE / 2.0f) * PATCH_SIZE / MESH_SIZE, 0.f,
                                             (z - MESH_SIZE / 2.0f) * PATCH_SIZE / MESH_SIZE);
            vdata[idx].texCoord = glm::vec2(static_cast<float>(x) / static_cast<float>(MESH_SIZE),
                                             static_cast<float>(z) / static_cast<float>(MESH_SIZE));
        }
    }

    std::vector<unsigned> idata;
    idata.reserve(static_cast<size_t>(MESH_SIZE) * static_cast<size_t>(MESH_SIZE) * 6u);
    for (int z = 0; z < MESH_SIZE; ++z) {
        for (int x = 0; x < MESH_SIZE; ++x) {
            int index = z * MESH_SIZE_1 + x;
            idata.push_back(static_cast<unsigned>(index));
            idata.push_back(static_cast<unsigned>(index + MESH_SIZE_1));
            idata.push_back(static_cast<unsigned>(index + MESH_SIZE_1 + 1));
            idata.push_back(static_cast<unsigned>(index));
            idata.push_back(static_cast<unsigned>(index + MESH_SIZE_1 + 1));
            idata.push_back(static_cast<unsigned>(index + 1));
        }
    }
    mIndexCount_ = static_cast<int>(idata.size());

    glGenVertexArrays(1, &mVao_);
    glBindVertexArray(mVao_);
    glGenBuffers(1, &mVbo_);
    glBindBuffer(GL_ARRAY_BUFFER, mVbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vdata.size() * sizeof(GridVertex)), vdata.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GridVertex), nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GridVertex), (void*)offsetof(GridVertex, texCoord));

    glGenBuffers(1, &mIbo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIbo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(idata.size() * sizeof(unsigned)), idata.data(),
                 GL_STATIC_DRAW);
    glBindVertexArray(0);
}

static void buildLodMesh(int meshSize, float patchSize, std::vector<GridVertex>& vertices,
                         std::vector<unsigned>& indices) {
    const int ms1 = meshSize + 1;
    vertices.resize(static_cast<size_t>(ms1 * ms1));
    for (int z = 0; z <= meshSize; ++z) {
        for (int x = 0; x <= meshSize; ++x) {
            int idx             = z * ms1 + x;
            vertices[idx].position = glm::vec3((x - meshSize / 2.0f) * patchSize / meshSize, 0.f,
                                                 (z - meshSize / 2.0f) * patchSize / meshSize);
            vertices[idx].texCoord = glm::vec2(static_cast<float>(x) / static_cast<float>(meshSize),
                                                static_cast<float>(z) / static_cast<float>(meshSize));
        }
    }
    indices.clear();
    for (int z = 0; z < meshSize; ++z) {
        for (int x = 0; x < meshSize; ++x) {
            int index = z * ms1 + x;
            indices.push_back(static_cast<unsigned>(index));
            indices.push_back(static_cast<unsigned>(index + ms1));
            indices.push_back(static_cast<unsigned>(index + ms1 + 1));
            indices.push_back(static_cast<unsigned>(index));
            indices.push_back(static_cast<unsigned>(index + ms1 + 1));
            indices.push_back(static_cast<unsigned>(index + 1));
        }
    }
}

void SimshipOcean::createLodMeshes_() {
    const int sizes[] = {256, 128, 32, 8, 4};
    for (int meshSize : sizes) {
        std::vector<GridVertex> v;
        std::vector<unsigned>   id;
        buildLodMesh(meshSize, static_cast<float>(SimshipOcean::PATCH_SIZE), v, id);

        GLuint vao = 0, vbo = 0, ibo = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(v.size() * sizeof(GridVertex)), v.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GridVertex), nullptr);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GridVertex), (void*)offsetof(GridVertex, texCoord));

        glGenBuffers(1, &ibo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(id.size() * sizeof(unsigned)), id.data(),
                     GL_STATIC_DRAW);

        glBindVertexArray(0);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ibo);

        lodVaos_.push_back(vao);
        lodIndexCounts_.push_back(static_cast<GLuint>(id.size()));
    }
}

bool SimshipOcean::buildShaders_() {
    const uint32_t log2n = highestSetBit(static_cast<uint32_t>(FFT_SIZE));
    char           defbuf[512];
    std::snprintf(defbuf, sizeof defbuf,
                  "#define FFT_SIZE %d\n"
                  "#define GRID_SIZE %d\n"
                  "#define LOG2_N_SIZE %u\n"
                  "#define PATCH_SIZE_X2_N %.9f\n",
                  FFT_SIZE, MESH_SIZE, log2n, static_cast<float>(PATCH_SIZE) * 2.f / static_cast<float>(FFT_SIZE));
    const std::string defs(defbuf);

    auto loadComp = [&](const char* fname) -> GLuint {
        std::string path   = shaderPath(fname);
        std::string fileSrc = loadTextFile(path);
        if (fileSrc.empty()) {
            std::fprintf(stderr, "simship_ocean: empty or missing shader %s\n", path.c_str());
            return 0;
        }

        // GLSL 要求 #version 必须是第一条指令；把宏定义插入到 #version 行之后。
        std::string src;
        const size_t eol = fileSrc.find('\n');
        if (fileSrc.rfind("#version", 0) == 0 && eol != std::string::npos) {
            src = fileSrc.substr(0, eol + 1) + defs + fileSrc.substr(eol + 1);
        } else {
            src = defs + fileSrc;
        }

        GLuint c = compileShader(GL_COMPUTE_SHADER, src.c_str());
        return linkCompute(c);
    };

    csSpectrum_      = loadComp("simship_ocean/updatespectrum.comp");
    csFft_           = loadComp("simship_ocean/fourier_fft.comp");
    csDisplacement_  = loadComp("simship_ocean/createdisplacement.comp");
    csGradients_     = loadComp("simship_ocean/creategradients.comp");

    std::string vsPath = shaderPath("simship_ocean/ocean.vert");
    std::string fsPath = shaderPath("simship_ocean/ocean.frag");
    std::string vsSrc  = loadTextFile(vsPath);
    std::string fsSrc  = loadTextFile(fsPath);
    if (vsSrc.empty() || fsSrc.empty()) {
        std::fprintf(stderr, "simship_ocean: missing ocean.vert/frag\n");
        return false;
    }
    GLuint v = compileShader(GL_VERTEX_SHADER, vsSrc.c_str());
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fsSrc.c_str());
    progOcean_ = linkProgramVsFs(v, f);

    if (!csSpectrum_ || !csFft_ || !csDisplacement_ || !csGradients_ || !progOcean_)
        return false;
    return true;
}

void SimshipOcean::fourier2D_(GLuint spectrumTex) {
    glBindImageTexture(0, spectrumTex, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, mTexTempData_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
    glUseProgram(csFft_);
    glDispatchCompute(FFT_SIZE, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glBindImageTexture(0, mTexTempData_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, spectrumTex, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
    glUseProgram(csFft_);
    glDispatchCompute(FFT_SIZE, 1, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SimshipOcean::uploadCoupledHeightfield(const Grid& g, float etaRef) {
    if (!initialized_ || !wakeEnabled_ || !mTexWake_ || g.NX <= 1 || g.NY <= 1 || g.dx <= 0.0f)
        return;

    const float halfW = 0.5f * static_cast<float>(g.NX) * g.dx;
    const float halfD = 0.5f * static_cast<float>(g.NY) * g.dx;
    std::vector<float> wake(static_cast<size_t>(FFT_SIZE) * static_cast<size_t>(FFT_SIZE), 0.0f);

    for (int y = 0; y < FFT_SIZE; ++y) {
        const float vz = (static_cast<float>(y) + 0.5f) / static_cast<float>(FFT_SIZE);
        const float wz = (vz - 0.5f) * (2.0f * halfD);
        const float gj = (wz + halfD) / g.dx;
        int         j  = static_cast<int>(std::round(gj));
        j = std::max(0, std::min(g.NY - 1, j));
        for (int x = 0; x < FFT_SIZE; ++x) {
            const float ux = (static_cast<float>(x) + 0.5f) / static_cast<float>(FFT_SIZE);
            const float wx = (ux - 0.5f) * (2.0f * halfW);
            const float gi = (wx + halfW) / g.dx;
            int         i  = static_cast<int>(std::round(gi));
            i = std::max(0, std::min(g.NX - 1, i));

            const float restDepth = std::max(0.0f, etaRef - g.B(i, j));
            const float hAnom     = g.H(i, j) - restDepth;
            wake[static_cast<size_t>(y) * static_cast<size_t>(FFT_SIZE) + static_cast<size_t>(x)] = hAnom;
        }
    }

    glBindTexture(GL_TEXTURE_2D, mTexWake_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, FFT_SIZE, FFT_SIZE, GL_RED, GL_FLOAT, wake.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SimshipOcean::update(float timeSeconds) {
    if (!initialized_)
        return;

    glUseProgram(csSpectrum_);
    glUniform1f(glGetUniformLocation(csSpectrum_, "time"), timeSeconds);
    glBindImageTexture(0, mTexInitialSpectrum_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, mTexFrequencies_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(2, mTexUpdatedSpectra_[0], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
    glBindImageTexture(3, mTexUpdatedSpectra_[1], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG32F);
    glDispatchCompute(FFT_SIZE / 16, FFT_SIZE / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    fourier2D_(mTexUpdatedSpectra_[0]);
    fourier2D_(mTexUpdatedSpectra_[1]);

    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(csDisplacement_);
    glUniform1f(glGetUniformLocation(csDisplacement_, "lambda"), lambdaChop_);
    glUniform1f(glGetUniformLocation(csDisplacement_, "wakeAmplitude"), wakeEnabled_ ? wakeAmplitude_ : 0.0f);
    glUniform1f(glGetUniformLocation(csDisplacement_, "wakeChop"), wakeEnabled_ ? wakeChop_ : 0.0f);
    glBindImageTexture(0, mTexUpdatedSpectra_[0], 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(1, mTexUpdatedSpectra_[1], 0, GL_TRUE, 0, GL_READ_ONLY, GL_RG32F);
    glBindImageTexture(2, mTexWake_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    glBindImageTexture(3, mTexDisplacements_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    glDispatchCompute(FFT_SIZE / 16, FFT_SIZE / 16, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    glUseProgram(csGradients_);
    glBindImageTexture(0, mTexDisplacements_, 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindImageTexture(1, mTexGradients_, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glDispatchCompute(FFT_SIZE / 16, FFT_SIZE / 16, 1);
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

    glUseProgram(0);
}

void SimshipOcean::render(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& eye,
                          const SimshipOceanLighting& L) {
    if (!initialized_ || lodVaos_.size() != 5)
        return;

    const glm::mat4 viewProj = proj * view;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(progOcean_);
    if (locMatViewProj_ >= 0)
        glUniformMatrix4fv(locMatViewProj_, 1, GL_FALSE, glm::value_ptr(viewProj));
    if (locEyePos_ >= 0)
        glUniform3fv(locEyePos_, 1, glm::value_ptr(eye));
    if (locOceanColor_ >= 0)
        glUniform3fv(locOceanColor_, 1, glm::value_ptr(oceanColor_));
    if (locTransparency_ >= 0)
        glUniform1f(locTransparency_, transparency_);
    if (locSunColor_ >= 0)
        glUniform3fv(locSunColor_, 1, glm::value_ptr(L.sunColor));
    if (locSunDir_ >= 0) {
        glm::vec3 sd = glm::normalize(L.sunDir);
        glUniform3fv(locSunDir_, 1, glm::value_ptr(sd));
    }
    if (locWaterLevel_ >= 0)
        glUniform1f(locWaterLevel_, L.waterLevel);
    if (locBUseScreenRefraction_ >= 0)
        glUniform1i(locBUseScreenRefraction_, L.useScreenRefraction ? 1 : 0);
    if (locClipNF_ >= 0)
        glUniform2f(locClipNF_, L.clipNear, L.clipFar);
    if (locDepthAbsorb_ >= 0)
        glUniform1f(locDepthAbsorb_, L.depthAbsorb);
    if (locViewport_ >= 0)
        glUniform4fv(locViewport_, 1, glm::value_ptr(L.viewport));
    if (locViewRot_ >= 0)
        glUniformMatrix3fv(locViewRot_, 1, GL_FALSE, glm::value_ptr(L.viewRot));
    if (locWaterIOR_ >= 0)
        glUniform1f(locWaterIOR_, L.ior);
    if (locWaterColor_ >= 0)
        glUniform3fv(locWaterColor_, 1, glm::value_ptr(L.waterColor));
    if (locWaterReflections_ >= 0)
        glUniform1f(locWaterReflections_, L.waterReflections);
    if (locRefractionMaxOffset_ >= 0)
        glUniform1f(locRefractionMaxOffset_, L.refractionMaxOffset);
    if (locRefractionLinTol_ >= 0)
        glUniform1f(locRefractionLinTol_, L.refractionLinTol);
    if (locBEnvmap_ >= 0)
        glUniform1i(locBEnvmap_, L.useEnvCubemap ? 1 : 0);
    if (locExposure_ >= 0)
        glUniform1f(locExposure_, L.exposure);
    if (locBAbsorbance_ >= 0)
        glUniform1i(locBAbsorbance_, L.absorbance ? 1 : 0);
    if (locAbsorbanceColor_ >= 0)
        glUniform3fv(locAbsorbanceColor_, 1, glm::value_ptr(L.absorbanceColor));
    if (locAbsorbanceCoeff_ >= 0)
        glUniform1f(locAbsorbanceCoeff_, L.absorbanceCoeff);
    if (locBShowPatch_ >= 0)
        glUniform1i(locBShowPatch_, debugShowPatch_ ? 1 : 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mTexDisplacements_);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, mTexGradients_);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, L.refractionTex);
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, L.sceneDepthTex);
    glActiveTexture(GL_TEXTURE0);

    glDepthMask(GL_FALSE);

    const int              nGrids = NbPatches;
    const float            margin = 1.1f * (PATCH_SIZE * 1.414f);
    std::vector<InstanceData> inst[5];

    const int cameraPatchX = static_cast<int>(std::round(eye.x / static_cast<float>(PATCH_SIZE)));
    const int cameraPatchZ = static_cast<int>(std::round(eye.z / static_cast<float>(PATCH_SIZE)));

    for (int j = -nGrids / 2; j <= nGrids / 2; ++j) {
        for (int i = -nGrids / 2; i <= nGrids / 2; ++i) {
            glm::vec3 center(PATCH_SIZE * (i + cameraPatchX), 0.f, PATCH_SIZE * (j + cameraPatchZ));
            glm::vec4 clip = viewProj * glm::vec4(center, 1.f);
            float     w    = clip.w;
            glm::vec3 ap(std::abs(clip.x), std::abs(clip.y), std::abs(clip.z));
            if (ap.x <= w + margin && ap.y <= w + margin) {
                float dist = glm::distance(center, eye);
                int   lod  = 4;
                if (dist < 600.f)
                    lod = 0;
                else if (dist < 1200.f)
                    lod = 1;
                else if (dist < 2400.f)
                    lod = 2;
                else if (dist < 4800.f)
                    lod = 3;
                InstanceData d;
                d.modelMatrix = glm::translate(glm::mat4(1.f), center);
                d.lod         = lod;
                inst[lod].push_back(d);
            }
        }
    }

    GLuint instanceVbo = 0;
    glGenBuffers(1, &instanceVbo);

    for (int lodLevel = 0; lodLevel < 5; ++lodLevel) {
        if (inst[lodLevel].empty())
            continue;
        glBindVertexArray(lodVaos_[static_cast<size_t>(lodLevel)]);
        glBindBuffer(GL_ARRAY_BUFFER, instanceVbo);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(inst[lodLevel].size() * sizeof(InstanceData)),
                     inst[lodLevel].data(), GL_DYNAMIC_DRAW);

        for (unsigned i = 0; i < 4; ++i) {
            glEnableVertexAttribArray(2 + i);
            glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData),
                                  (void*)(sizeof(glm::vec4) * i));
            glVertexAttribDivisor(2 + i, 1);
        }
        glEnableVertexAttribArray(6);
        glVertexAttribIPointer(6, 1, GL_INT, sizeof(InstanceData), (void*)offsetof(InstanceData, lod));
        glVertexAttribDivisor(6, 1);

        glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(lodIndexCounts_[static_cast<size_t>(lodLevel)]),
                                GL_UNSIGNED_INT, nullptr, static_cast<GLsizei>(inst[lodLevel].size()));
    }

    glDeleteBuffers(1, &instanceVbo);

    for (unsigned i = 2; i <= 6; ++i) {
        glDisableVertexAttribArray(i);
        glVertexAttribDivisor(i, 0);
    }
    glBindVertexArray(0);
    glUseProgram(0);
    glDepthMask(GL_TRUE);
}
