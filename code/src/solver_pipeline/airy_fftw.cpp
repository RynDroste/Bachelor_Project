#include "solver_pipeline/airy_fftw.h"

#include "solver_pipeline/airy_cuda_kernels.hpp"

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

namespace {

#define CUDA_CHECK(x)                                                                                                  \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));                \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define CUFFT_CHECK(x)                                                                                                 \
    do {                                                                                                               \
        cufftResult _r = (x);                                                                                          \
        if (_r != CUFFT_SUCCESS) {                                                                                     \
            std::fprintf(stderr, "%s:%d cuFFT error: %d\n", __FILE__, __LINE__, static_cast<int>(_r));               \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define CUDA_POST_KERNEL() CUDA_CHECK(cudaGetLastError())

constexpr float kDepths[4] = {1.f, 4.f, 16.f, 64.f};

} // namespace

struct AiryEWaveFFTW::Impl {
    int   nx = 0;
    int   ny = 0;
    float dx = 1.f;

    float*         d_real = nullptr;
    cufftComplex*  d_hatH = nullptr;
    cufftComplex*  d_hatQx = nullptr;
    cufftComplex*  d_hatQy = nullptr;
    cufftComplex*  d_work = nullptr;
    float*         d_spatial = nullptr;
    float*         d_h_bar = nullptr;
    float*         d_qxc = nullptr;
    float*         d_qyc = nullptr;

    cufftHandle planFwd = 0;
    cufftHandle planInv = 0;

    ~Impl() {
        if (planFwd)
            cufftDestroy(planFwd);
        if (planInv)
            cufftDestroy(planInv);
        cudaFree(d_real);
        cudaFree(d_hatH);
        cudaFree(d_hatQx);
        cudaFree(d_hatQy);
        cudaFree(d_work);
        cudaFree(d_spatial);
        cudaFree(d_h_bar);
        cudaFree(d_qxc);
        cudaFree(d_qyc);
    }
};

AiryEWaveFFTW::AiryEWaveFFTW(int nx, int ny, float dx)
    : nx_(nx)
    , ny_(ny)
    , dx_(dx)
    , impl_(std::make_unique<Impl>()) {
    CUDA_CHECK(cudaSetDevice(0));

    Impl& im   = *impl_;
    im.nx      = nx;
    im.ny      = ny;
    im.dx      = dx;
    const size_t n   = static_cast<size_t>(nx) * static_cast<size_t>(ny);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_real), n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_hatH), n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_hatQx), n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_hatQy), n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_work), n * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_spatial), 8u * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_h_bar), n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_qxc), n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&im.d_qyc), n * sizeof(float)));

    CUFFT_CHECK(cufftPlan2d(&im.planFwd, nx, ny, CUFFT_C2C));
    CUFFT_CHECK(cufftPlan2d(&im.planInv, nx, ny, CUFFT_C2C));
}

AiryEWaveFFTW::~AiryEWaveFFTW() = default;

void AiryEWaveFFTW::step(float dt, float g,
                         const float* h_tilde_sym,
                         const float* h_bar,
                         float*       qx_tilde,
                         float*       qy_tilde) {
    Impl&       im   = *impl_;
    const int   nx   = nx_;
    const int   ny   = ny_;
    const float dx   = dx_;
    const float dy   = dx;
    const int   n    = nx * ny;
    const float invN = 1.f / static_cast<float>(n);
    const size_t nBytesF = static_cast<size_t>(n) * sizeof(float);

    float* d_sx0 = im.d_spatial + 0u * static_cast<size_t>(n);
    float* d_sx1 = im.d_spatial + 1u * static_cast<size_t>(n);
    float* d_sx2 = im.d_spatial + 2u * static_cast<size_t>(n);
    float* d_sx3 = im.d_spatial + 3u * static_cast<size_t>(n);
    float* d_sy0 = im.d_spatial + 4u * static_cast<size_t>(n);
    float* d_sy1 = im.d_spatial + 5u * static_cast<size_t>(n);
    float* d_sy2 = im.d_spatial + 6u * static_cast<size_t>(n);
    float* d_sy3 = im.d_spatial + 7u * static_cast<size_t>(n);

    CUDA_CHECK(cudaMemcpy(im.d_qxc, qx_tilde, nBytesF, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_qyc, qy_tilde, nBytesF, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(im.d_real, h_tilde_sym, nBytesF, cudaMemcpyHostToDevice));
    airy_cuda_launch_real_to_complex(im.d_real, im.d_hatH, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatH, im.d_hatH, CUFFT_FORWARD));

    airy_cuda_launch_real_to_complex(im.d_qxc, im.d_hatQx, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatQx, im.d_hatQx, CUFFT_FORWARD));

    airy_cuda_launch_real_to_complex(im.d_qyc, im.d_hatQy, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatQy, im.d_hatQy, CUFFT_FORWARD));

    for (int d = 0; d < 4; ++d) {
        const float hDepth = kDepths[d];
        float* d_sx = im.d_spatial + static_cast<size_t>(d) * static_cast<size_t>(n);
        float* d_sy = im.d_spatial + (4u + static_cast<size_t>(d)) * static_cast<size_t>(n);

        airy_cuda_launch_spectral_qx(im.d_hatH, im.d_hatQx, im.d_work, nx, ny, dx, dy, dt, g, hDepth, 0);
        CUDA_POST_KERNEL();
        CUFFT_CHECK(cufftExecC2C(im.planInv, im.d_work, im.d_work, CUFFT_INVERSE));
        airy_cuda_launch_cpx_to_real_scaled(im.d_work, d_sx, invN, n, 0);
        CUDA_POST_KERNEL();

        airy_cuda_launch_spectral_qy(im.d_hatH, im.d_hatQy, im.d_work, nx, ny, dx, dy, dt, g, hDepth, 0);
        CUDA_POST_KERNEL();
        CUFFT_CHECK(cufftExecC2C(im.planInv, im.d_work, im.d_work, CUFFT_INVERSE));
        airy_cuda_launch_cpx_to_real_scaled(im.d_work, d_sy, invN, n, 0);
        CUDA_POST_KERNEL();
    }

    CUDA_CHECK(cudaMemcpy(im.d_h_bar, h_bar, nBytesF, cudaMemcpyHostToDevice));
    airy_cuda_launch_blend(d_sx0, d_sx1, d_sx2, d_sx3, d_sy0, d_sy1, d_sy2, d_sy3, im.d_h_bar, im.d_qxc, im.d_qyc,
                           nx, ny, 0);
    CUDA_POST_KERNEL();

    CUDA_CHECK(cudaMemcpy(qx_tilde, im.d_qxc, nBytesF, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(qy_tilde, im.d_qyc, nBytesF, cudaMemcpyDeviceToHost));
}

void AiryEWaveFFTW::stepDevice(float dt, float g_grav, const float* d_h_tilde_sym, const float* d_h_bar,
                               float* d_qx_tilde, float* d_qy_tilde) {
    CUDA_CHECK(cudaSetDevice(0));
    Impl&       im   = *impl_;
    const int   nx   = nx_;
    const int   ny   = ny_;
    const float dx   = dx_;
    const float dy   = dx;
    const int   n    = nx * ny;
    const float invN = 1.f / static_cast<float>(n);
    const size_t nBytesF = static_cast<size_t>(n) * sizeof(float);

    float* d_sx0 = im.d_spatial + 0u * static_cast<size_t>(n);
    float* d_sx1 = im.d_spatial + 1u * static_cast<size_t>(n);
    float* d_sx2 = im.d_spatial + 2u * static_cast<size_t>(n);
    float* d_sx3 = im.d_spatial + 3u * static_cast<size_t>(n);
    float* d_sy0 = im.d_spatial + 4u * static_cast<size_t>(n);
    float* d_sy1 = im.d_spatial + 5u * static_cast<size_t>(n);
    float* d_sy2 = im.d_spatial + 6u * static_cast<size_t>(n);
    float* d_sy3 = im.d_spatial + 7u * static_cast<size_t>(n);

    CUDA_CHECK(cudaMemcpy(im.d_qxc, d_qx_tilde, nBytesF, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(im.d_qyc, d_qy_tilde, nBytesF, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaMemcpy(im.d_real, d_h_tilde_sym, nBytesF, cudaMemcpyDeviceToDevice));
    airy_cuda_launch_real_to_complex(im.d_real, im.d_hatH, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatH, im.d_hatH, CUFFT_FORWARD));

    airy_cuda_launch_real_to_complex(im.d_qxc, im.d_hatQx, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatQx, im.d_hatQx, CUFFT_FORWARD));

    airy_cuda_launch_real_to_complex(im.d_qyc, im.d_hatQy, n, 0);
    CUDA_POST_KERNEL();
    CUFFT_CHECK(cufftExecC2C(im.planFwd, im.d_hatQy, im.d_hatQy, CUFFT_FORWARD));

    for (int d = 0; d < 4; ++d) {
        float* d_sx = im.d_spatial + static_cast<size_t>(d) * static_cast<size_t>(n);
        float* d_sy = im.d_spatial + (4u + static_cast<size_t>(d)) * static_cast<size_t>(n);

        airy_cuda_launch_spectral_qx(im.d_hatH, im.d_hatQx, im.d_work, nx, ny, dx, dy, dt, g_grav, kDepths[d], 0);
        CUDA_POST_KERNEL();
        CUFFT_CHECK(cufftExecC2C(im.planInv, im.d_work, im.d_work, CUFFT_INVERSE));
        airy_cuda_launch_cpx_to_real_scaled(im.d_work, d_sx, invN, n, 0);
        CUDA_POST_KERNEL();

        airy_cuda_launch_spectral_qy(im.d_hatH, im.d_hatQy, im.d_work, nx, ny, dx, dy, dt, g_grav, kDepths[d], 0);
        CUDA_POST_KERNEL();
        CUFFT_CHECK(cufftExecC2C(im.planInv, im.d_work, im.d_work, CUFFT_INVERSE));
        airy_cuda_launch_cpx_to_real_scaled(im.d_work, d_sy, invN, n, 0);
        CUDA_POST_KERNEL();
    }

    CUDA_CHECK(cudaMemcpy(im.d_h_bar, d_h_bar, nBytesF, cudaMemcpyDeviceToDevice));
    airy_cuda_launch_blend(d_sx0, d_sx1, d_sx2, d_sx3, d_sy0, d_sy1, d_sy2, d_sy3, im.d_h_bar, im.d_qxc, im.d_qyc,
                           nx, ny, 0);
    CUDA_POST_KERNEL();

    CUDA_CHECK(cudaMemcpy(d_qx_tilde, im.d_qxc, nBytesF, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_qy_tilde, im.d_qyc, nBytesF, cudaMemcpyDeviceToDevice));
}
