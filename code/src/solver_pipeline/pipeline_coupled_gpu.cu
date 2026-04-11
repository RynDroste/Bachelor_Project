#include "solver_pipeline/pipeline_coupled_gpu.hpp"

#include "solver_pipeline/airy_fftw.h"
#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/transport_gpu.hpp"
#include "solver_pipeline/wave_decompose_gpu.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

#define CP_CUDA_OK(x)                                                                                                  \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

__global__ void sym_htilde_k(float* __restrict__ out, const float* __restrict__ cur, const float* __restrict__ prev,
                             int n, int have_prev) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = have_prev ? 0.5f * (prev[i] + cur[i]) : cur[i];
}

__global__ void add_vec_k(float* __restrict__ out, const float* __restrict__ a, const float* __restrict__ b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    out[i] = a[i] + b[i];
}

inline int blocks1d(int n, int t) { return (n + t - 1) / t; }

struct CoupledScratch {
    int    nx = 0;
    int    ny = 0;
    float* d_bar0_h    = nullptr;
    float* d_bar0_qx   = nullptr;
    float* d_bar0_qy   = nullptr;
    float* d_bar1_h    = nullptr;
    float* d_bar1_qx   = nullptr;
    float* d_bar1_qy   = nullptr;
    float* d_g_h       = nullptr;
    float* d_g_qx      = nullptr;
    float* d_g_qy      = nullptr;
    float* d_h_sym     = nullptr;
    float* d_h_prev    = nullptr;

    void freeAll() {
        cudaFree(d_bar0_h);
        cudaFree(d_bar0_qx);
        cudaFree(d_bar0_qy);
        cudaFree(d_bar1_h);
        cudaFree(d_bar1_qx);
        cudaFree(d_bar1_qy);
        cudaFree(d_g_h);
        cudaFree(d_g_qx);
        cudaFree(d_g_qy);
        cudaFree(d_h_sym);
        cudaFree(d_h_prev);
        d_bar0_h = d_bar0_qx = d_bar0_qy = d_bar1_h = d_bar1_qx = d_bar1_qy = nullptr;
        d_g_h = d_g_qx = d_g_qy = d_h_sym = d_h_prev = nullptr;
        nx = ny = 0;
    }

    ~CoupledScratch() { freeAll(); }

    void ensure(int nx_, int ny_) {
        if (nx_ == nx && ny_ == ny && d_bar0_h)
            return;
        freeAll();
        nx = nx_;
        ny = ny_;
        const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
        const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar0_h), ncell * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar0_qx), nqx * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar0_qy), nqy * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar1_h), ncell * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar1_qx), nqx * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_bar1_qy), nqy * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_g_h), ncell * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_g_qx), nqx * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_g_qy), nqy * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_h_sym), ncell * sizeof(float)));
        CP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_h_prev), ncell * sizeof(float)));
    }
};

CoupledScratch g_cp;

} // namespace

void coupledSubstepGpu(Grid& g, float halfW, float halfD, WaveDecomposition& dec, AiryEWaveFFTW& airy,
                       std::vector<float>& hTildeSym, std::vector<float>& hTildePrevHalf, bool& haveHtildePrevHalf,
                       float gradPenaltyD, float transportGamma, int waveDiffuseIters) {
    (void)dec;
    (void)hTildeSym;
    (void)hTildePrevHalf;
    CP_CUDA_OK(cudaSetDevice(0));

    const int   nx    = g.NX;
    const int   ny    = g.NY;
    const float dx    = g.dx;
    const float dt    = g.dt;
    const int   ncell = nx * ny;
    const int   nqx   = (nx + 1) * ny;
    const int   nqy   = nx * (ny + 1);

    g_cp.ensure(nx, ny);

    swePrefetchDeviceTerrain(g);

    const WaveDecompGpuPtrs wd = waveDecomposeGpuDeviceOnly(g, gradPenaltyD, waveDiffuseIters);

    // DEBUG: print h_tilde vs h_bar RMS every 120 steps
    {
        static int s_dbg_counter = 0;
        if (++s_dbg_counter % 120 == 0) {
            std::vector<float> h_tilde_host(ncell), h_bar_host(ncell);
            cudaMemcpy(h_tilde_host.data(), wd.d_h_tilde, ncell * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_bar_host.data(),   wd.d_h_bar,   ncell * sizeof(float), cudaMemcpyDeviceToHost);
            double rms_tilde = 0, rms_bar = 0;
            for (int i = 0; i < ncell; ++i) {
                rms_tilde += h_tilde_host[i] * h_tilde_host[i];
                rms_bar   += h_bar_host[i]   * h_bar_host[i];
            }
            rms_tilde = std::sqrt(rms_tilde / ncell);
            rms_bar   = std::sqrt(rms_bar   / ncell);
            std::printf("[coupled] step=%d  h_tilde RMS=%.6f  h_bar RMS=%.6f  ratio=%.4f\n",
                        s_dbg_counter, rms_tilde, rms_bar,
                        rms_bar > 1e-12 ? rms_tilde / rms_bar : 0.0);
        }
    }

    const size_t szCell = static_cast<size_t>(ncell) * sizeof(float);
    const size_t szQx   = static_cast<size_t>(nqx) * sizeof(float);
    const size_t szQy   = static_cast<size_t>(nqy) * sizeof(float);

    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar0_h, wd.d_h_bar, szCell, cudaMemcpyDeviceToDevice));
    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar0_qx, wd.d_qx_bar, szQx, cudaMemcpyDeviceToDevice));
    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar0_qy, wd.d_qy_bar, szQy, cudaMemcpyDeviceToDevice));
    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar1_h, wd.d_h_bar, szCell, cudaMemcpyDeviceToDevice));
    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar1_qx, wd.d_qx_bar, szQx, cudaMemcpyDeviceToDevice));
    CP_CUDA_OK(cudaMemcpy(g_cp.d_bar1_qy, wd.d_qy_bar, szQy, cudaMemcpyDeviceToDevice));

    sweStepGpuInPlaceDevice(g_cp.d_bar1_h, g_cp.d_bar1_qx, g_cp.d_bar1_qy, g, nx, ny, dx, dt);

    constexpr int t256 = 256;
    sym_htilde_k<<<blocks1d(ncell, t256), t256>>>(g_cp.d_h_sym, wd.d_h_tilde, g_cp.d_h_prev, ncell,
                                                  haveHtildePrevHalf ? 1 : 0);
    CP_CUDA_OK(cudaGetLastError());
    CP_CUDA_OK(cudaDeviceSynchronize());

    airy.stepDevice(dt, 9.81f, g_cp.d_h_sym, wd.d_h_bar, wd.d_qx_tilde, wd.d_qy_tilde);
    CP_CUDA_OK(cudaDeviceSynchronize());

    transportSurfaceGpuDevice(g_cp.d_bar0_h, g_cp.d_bar0_qx, g_cp.d_bar0_qy, g_cp.d_bar1_h, g_cp.d_bar1_qx,
                              g_cp.d_bar1_qy, wd.d_h_tilde, wd.d_qx_tilde, wd.d_qy_tilde, nx, ny, dx, halfW, halfD, dt,
                              transportGamma);

    CP_CUDA_OK(cudaMemcpy(g_cp.d_h_prev, wd.d_h_tilde, szCell, cudaMemcpyDeviceToDevice));
    haveHtildePrevHalf = true;

    add_vec_k<<<blocks1d(ncell, t256), t256>>>(g_cp.d_g_h, g_cp.d_bar1_h, wd.d_h_tilde, ncell);
    add_vec_k<<<blocks1d(nqx, t256), t256>>>(g_cp.d_g_qx, g_cp.d_bar1_qx, wd.d_qx_tilde, nqx);
    add_vec_k<<<blocks1d(nqy, t256), t256>>>(g_cp.d_g_qy, g_cp.d_bar1_qy, wd.d_qy_tilde, nqy);
    CP_CUDA_OK(cudaGetLastError());
    CP_CUDA_OK(cudaDeviceSynchronize());

    sweApplyBoundaryGpuInPlaceDevice(g_cp.d_g_h, g_cp.d_g_qx, g_cp.d_g_qy, g, nx, ny, dx, dt);

    sweDownloadGridFromDevice(g, g_cp.d_g_h, g_cp.d_g_qx, g_cp.d_g_qy);
}
