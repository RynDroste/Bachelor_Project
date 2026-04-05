#include "solver_pipeline/transport_gpu.hpp"

#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/wavedecomposer.h"

#include <cstdio>
#include <cuda_runtime.h>

#define BP_JT_CUDA_OK(x)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace bp_jt_detail {

#define JT_CUDA_CHECK(x)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define JT_POST_KERNEL() JT_CUDA_CHECK(cudaGetLastError())

constexpr float kDryEps = 1e-4f;

__device__ inline float d_faceH_X(const float* h, const float* qx, int nx, int ny, int i, int j) {
    const int il = max(0, i - 1);
    const int ir = min(nx - 1, i);
    const float qxf = qx[i + j * (nx + 1)];
    return (qxf >= 0.f) ? h[il + j * nx] : h[ir + j * nx];
}

__device__ inline float d_faceH_Y(const float* h, const float* qy, int nx, int ny, int i, int j) {
    const int jd = max(0, j - 1);
    const int ju = min(ny - 1, j);
    const float qyf = qy[i + j * nx];
    return (qyf >= 0.f) ? h[i + jd * nx] : h[i + ju * nx];
}

__device__ inline float d_uFaceX(const float* h, const float* qx, int nx, int ny, int i, int j) {
    const float hf = d_faceH_X(h, qx, nx, ny, i, j);
    const float qxf = qx[i + j * (nx + 1)];
    return (hf < kDryEps) ? 0.f : qxf / hf;
}

__device__ inline float d_uFaceY(const float* h, const float* qy, int nx, int ny, int i, int j) {
    const float hf = d_faceH_Y(h, qy, nx, ny, i, j);
    const float qyf = qy[i + j * nx];
    return (hf < kDryEps) ? 0.f : qyf / hf;
}

__device__ inline float d_uXmid(const float* h0, const float* qx0, const float* h1, const float* qx1,
                               int nx, int ny, int i, int j) {
    return 0.5f * (d_uFaceX(h0, qx0, nx, ny, i, j) + d_uFaceX(h1, qx1, nx, ny, i, j));
}

__device__ inline float d_uYmid(const float* h0, const float* qy0, const float* h1, const float* qy1,
                                int nx, int ny, int i, int j) {
    return 0.5f * (d_uFaceY(h0, qy0, nx, ny, i, j) + d_uFaceY(h1, qy1, nx, ny, i, j));
}

__device__ inline float d_divUmidCell(const float* ha, const float* qxa, const float* qya,
                                      const float* hb, const float* qxb, const float* qyb,
                                      int nx, int ny, int i, int j, float dx) {
    const float uxR = d_uXmid(ha, qxa, hb, qxb, nx, ny, i + 1, j);
    const float uxL = d_uXmid(ha, qxa, hb, qxb, nx, ny, i, j);
    const float uyT = d_uYmid(ha, qya, hb, qyb, nx, ny, i, j + 1);
    const float uyB = d_uYmid(ha, qya, hb, qyb, nx, ny, i, j);
    return (uxR - uxL + uyT - uyB) / dx;
}

__device__ inline float d_divUCellBar1(const float* h, const float* qx, const float* qy,
                                        int nx, int ny, int i, int j, float dx) {
    const float uxR = d_uFaceX(h, qx, nx, ny, i + 1, j);
    const float uxL = d_uFaceX(h, qx, nx, ny, i, j);
    const float uyT = d_uFaceY(h, qy, nx, ny, i, j + 1);
    const float uyB = d_uFaceY(h, qy, nx, ny, i, j);
    return (uxR - uxL + uyT - uyB) / dx;
}

__device__ inline float d_GofDiv(float div, float gamma) { return fminf(-div, -gamma * div); }

__device__ inline float d_clampExpArg(float x) { return fminf(fmaxf(x, -30.f), 30.f); }

__device__ inline float d_uyM_at(const float* ha, const float* qya, const float* hb, const float* qyb,
                                 int nx, int ny, int ii, int jj) {
    ii = max(0, min(nx - 1, ii));
    jj = max(0, min(ny, jj));
    return d_uYmid(ha, qya, hb, qyb, nx, ny, ii, jj);
}

__device__ inline float d_uzAtQxFace(const float* ha, const float* qxa, const float* qya,
                                     const float* hb, const float* qxb, const float* qyb,
                                     int nx, int ny, int i, int j) {
    const int im = max(0, i - 1);
    const int ip = min(nx - 1, i);
    const int j0 = max(0, min(ny - 1, j));
    const int j1 = min(j + 1, ny);
    return 0.25f * (d_uyM_at(ha, qya, hb, qyb, nx, ny, im, j0) + d_uyM_at(ha, qya, hb, qyb, nx, ny, ip, j0) +
                    d_uyM_at(ha, qya, hb, qyb, nx, ny, im, j1) + d_uyM_at(ha, qya, hb, qyb, nx, ny, ip, j1));
}

__device__ inline float d_uxM_at(const float* ha, const float* qxa, const float* hb, const float* qxb,
                                 int nx, int ny, int ii, int jj) {
    ii = max(0, min(nx, ii));
    jj = max(0, min(ny - 1, jj));
    return d_uXmid(ha, qxa, hb, qxb, nx, ny, ii, jj);
}

__device__ inline float d_uxAtQyFace(const float* ha, const float* qxa, const float* qya,
                                       const float* hb, const float* qxb, const float* qyb,
                                       int nx, int ny, int i, int j) {
    const int jm = max(0, j - 1);
    const int jp = min(ny - 1, j);
    const int i0 = max(0, min(nx - 1, i));
    const int i1 = min(i + 1, nx);
    return 0.25f * (d_uxM_at(ha, qxa, hb, qxb, nx, ny, i0, jm) + d_uxM_at(ha, qxa, hb, qxb, nx, ny, i1, jm) +
                    d_uxM_at(ha, qxa, hb, qxb, nx, ny, i0, jp) + d_uxM_at(ha, qxa, hb, qxb, nx, ny, i1, jp));
}

__device__ inline float d_sampleQx(const float* q, int nx, int ny, float fi, float fj) {
    fi = fminf(fmaxf(fi, 0.f), static_cast<float>(nx));
    fj = fminf(fmaxf(fj, 0.f), static_cast<float>(ny - 1) - 1e-5f);
    int i0 = static_cast<int>(floorf(fi));
    int j0 = static_cast<int>(floorf(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = max(0, min(nx - 1, i0));
    j0 = max(0, min(ny - 1, j0));
    const int i1 = min(i0 + 1, nx);
    const int j1 = min(j0 + 1, ny - 1);
    const float q00 = q[i0 + j0 * (nx + 1)];
    const float q10 = q[i1 + j0 * (nx + 1)];
    const float q01 = q[i0 + j1 * (nx + 1)];
    const float q11 = q[i1 + j1 * (nx + 1)];
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

__device__ inline float d_sampleQy(const float* q, int nx, int ny, float fi, float fj) {
    fi = fminf(fmaxf(fi, 0.f), static_cast<float>(nx - 1) - 1e-5f);
    fj = fminf(fmaxf(fj, 0.f), static_cast<float>(ny));
    int i0 = static_cast<int>(floorf(fi));
    int j0 = static_cast<int>(floorf(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = max(0, min(nx - 2, i0));
    j0 = max(0, min(ny - 1, j0));
    const int i1 = min(i0 + 1, nx - 1);
    const int j1 = min(j0 + 1, ny);
    const float q00 = q[i0 + j0 * nx];
    const float q10 = q[i1 + j0 * nx];
    const float q01 = q[i0 + j1 * nx];
    const float q11 = q[i1 + j1 * nx];
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

__device__ inline float d_sampleHcell(const float* h, int nx, int ny, float fi, float fj) {
    fi = fminf(fmaxf(fi, 0.f), static_cast<float>(nx - 1) - 1e-5f);
    fj = fminf(fmaxf(fj, 0.f), static_cast<float>(ny - 1) - 1e-5f);
    int i0 = static_cast<int>(floorf(fi));
    int j0 = static_cast<int>(floorf(fj));
    const float tx = fi - static_cast<float>(i0);
    const float ty = fj - static_cast<float>(j0);
    i0 = max(0, min(nx - 2, i0));
    j0 = max(0, min(ny - 2, j0));
    const float q00 = h[i0 + j0 * nx];
    const float q10 = h[i0 + 1 + j0 * nx];
    const float q01 = h[i0 + (j0 + 1) * nx];
    const float q11 = h[i0 + 1 + (j0 + 1) * nx];
    const float q0  = q00 + tx * (q10 - q00);
    const float q1  = q01 + tx * (q11 - q01);
    return q0 + ty * (q1 - q0);
}

// -----------------------------------------------------------------------------

__global__ void transport_qx_damp_k(float* qx_tilde, const float* ha, const float* qxa, const float* qya,
                                     const float* hb, const float* qxb, const float* qyb, int nx, int ny,
                                     float dx, float gamma, float dt) {
    const int nqx = (nx + 1) * ny;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nqx)
        return;
    const int i = tid % (nx + 1);
    const int j = tid / (nx + 1);

    float divC;
    if (i <= 0)
        divC = d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, 0, j, dx);
    else if (i >= nx)
        divC = d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, nx - 1, j, dx);
    else
        divC = 0.5f * (d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i - 1, j, dx) +
                       d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, j, dx));
    const float G = d_GofDiv(divC, gamma);
    qx_tilde[tid] *= expf(d_clampExpArg(G * dt));
}

__global__ void transport_qy_damp_k(float* qy_tilde, const float* ha, const float* qxa, const float* qya,
                                       const float* hb, const float* qxb, const float* qyb, int nx, int ny,
                                       float dx, float gamma, float dt) {
    const int nqy = nx * (ny + 1);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nqy)
        return;
    const int i = tid % nx;
    const int j = tid / nx;

    float divC;
    if (j <= 0)
        divC = d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, 0, dx);
    else if (j >= ny)
        divC = d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, ny - 1, dx);
    else
        divC = 0.5f * (d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, j - 1, dx) +
                       d_divUmidCell(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, j, dx));
    const float G = d_GofDiv(divC, gamma);
    qy_tilde[tid] *= expf(d_clampExpArg(G * dt));
}

__global__ void transport_qx_advect_k(float* qx_out, const float* qx_src, const float* ha, const float* qxa,
                                         const float* qya, const float* hb, const float* qxb, const float* qyb,
                                         int nx, int ny, float dx, float halfW, float halfD, float dt) {
    const int nqx = (nx + 1) * ny;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nqx)
        return;
    const int i = tid % (nx + 1);
    const int j = tid / (nx + 1);

    const float x  = static_cast<float>(i) * dx - halfW;
    const float z  = (static_cast<float>(j) + 0.5f) * dx - halfD;
    const float ux = d_uXmid(ha, qxa, hb, qxb, nx, ny, i, j);
    const float uz = d_uzAtQxFace(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, j);
    const float xd = x - ux * dt;
    const float zd = z - uz * dt;
    const float fi = (xd + halfW) / dx;
    const float fj = (zd + halfD) / dx - 0.5f;
    qx_out[tid]    = d_sampleQx(qx_src, nx, ny, fi, fj);
}

__global__ void transport_qy_advect_k(float* qy_out, const float* qy_src, const float* ha, const float* qxa,
                                       const float* qya, const float* hb, const float* qxb, const float* qyb, int nx,
                                       int ny, float dx, float halfW, float halfD, float dt) {
    const int nqy = nx * (ny + 1);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nqy)
        return;
    const int i = tid % nx;
    const int j = tid / nx;

    const float x  = (static_cast<float>(i) + 0.5f) * dx - halfW;
    const float z  = static_cast<float>(j) * dx - halfD;
    const float uz = d_uYmid(ha, qya, hb, qyb, nx, ny, i, j);
    const float ux = d_uxAtQyFace(ha, qxa, qya, hb, qxb, qyb, nx, ny, i, j);
    const float xd = x - ux * dt;
    const float zd = z - uz * dt;
    const float fi = (xd + halfW) / dx - 0.5f;
    const float fj = (zd + halfD) / dx;
    qy_out[tid]    = d_sampleQy(qy_src, nx, ny, fi, fj);
}

__global__ void transport_h_damp_k(float* h_tilde, const float* h1, const float* qx1, const float* qy1, int nx,
                                      int ny, float dx, float gamma, float dt) {
    const int ncell = nx * ny;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ncell)
        return;
    const int i = tid % nx;
    const int j = tid / nx;

    const float divC = d_divUCellBar1(h1, qx1, qy1, nx, ny, i, j, dx);
    const float G    = d_GofDiv(divC, gamma);
    h_tilde[tid] *= expf(d_clampExpArg(G * dt));
}

__global__ void transport_h_advect_k(float* h_out, const float* h_src, const float* h1, const float* qx1,
                                        const float* qy1, int nx, int ny, float dx, float halfW, float halfD,
                                        float dt) {
    const int ncell = nx * ny;
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ncell)
        return;
    const int i = tid % nx;
    const int j = tid / nx;

    const float x = (static_cast<float>(i) + 0.5f) * dx - halfW;
    const float z = (static_cast<float>(j) + 0.5f) * dx - halfD;
    const float ux =
        0.5f * (d_uFaceX(h1, qx1, nx, ny, i, j) + d_uFaceX(h1, qx1, nx, ny, i + 1, j));
    const float uz =
        0.5f * (d_uFaceY(h1, qy1, nx, ny, i, j) + d_uFaceY(h1, qy1, nx, ny, i, j + 1));
    const float xd = x - ux * dt;
    const float zd = z - uz * dt;
    const float fi = (xd + halfW) / dx - 0.5f;
    const float fj = (zd + halfD) / dx - 0.5f;
    h_out[tid]     = d_sampleHcell(h_src, nx, ny, fi, fj);
}

inline int jt_blocks_for(int n, int threads) { return (n + threads - 1) / threads; }

struct JtGpuScratch {
    float* d_h0        = nullptr;
    float* d_qx0       = nullptr;
    float* d_qy0       = nullptr;
    float* d_h1        = nullptr;
    float* d_qx1       = nullptr;
    float* d_qy1       = nullptr;
    float* d_qx_tilde  = nullptr;
    float* d_qy_tilde  = nullptr;
    float* d_h_tilde   = nullptr;
    float* d_qx_saved  = nullptr;
    float* d_qy_saved  = nullptr;
    float* d_h_saved   = nullptr;
    int    nx          = 0;
    int    ny          = 0;

    void freeAll() {
        cudaFree(d_h0);
        cudaFree(d_qx0);
        cudaFree(d_qy0);
        cudaFree(d_h1);
        cudaFree(d_qx1);
        cudaFree(d_qy1);
        cudaFree(d_qx_tilde);
        cudaFree(d_qy_tilde);
        cudaFree(d_h_tilde);
        cudaFree(d_qx_saved);
        cudaFree(d_qy_saved);
        cudaFree(d_h_saved);
        d_h0 = d_qx0 = d_qy0 = d_h1 = d_qx1 = d_qy1 = nullptr;
        d_qx_tilde = d_qy_tilde = d_h_tilde = d_qx_saved = d_qy_saved = d_h_saved = nullptr;
        nx = ny = 0;
    }

    ~JtGpuScratch() { freeAll(); }

    void ensure(int nx_, int ny_) {
        if (nx_ == nx && ny_ == ny)
            return;
        freeAll();
        nx = nx_;
        ny = ny_;
        const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
        const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);

        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h0), ncell * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx0), nqx * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy0), nqy * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h1), ncell * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx1), nqx * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy1), nqy * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_tilde), nqx * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_tilde), nqy * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_tilde), ncell * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_saved), nqx * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_saved), nqy * sizeof(float)));
        JT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_saved), ncell * sizeof(float)));
    }
};

JtGpuScratch g_jt;

void runTransportKernels(int nx, int ny, float dx, float halfW, float halfD, float dt, float gamma) {
    constexpr int threads = 256;
    const size_t  ncell   = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t  nqx     = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    const size_t  nqy     = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);
    const int     nqxN    = static_cast<int>(nqx);
    const int     nqyN    = static_cast<int>(nqy);
    const int     ncellN  = static_cast<int>(ncell);

    transport_qx_damp_k<<<jt_blocks_for(nqxN, threads), threads>>>(
        g_jt.d_qx_tilde, g_jt.d_h0, g_jt.d_qx0, g_jt.d_qy0, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny, dx, gamma, dt);
    JT_POST_KERNEL();

    transport_qy_damp_k<<<jt_blocks_for(nqyN, threads), threads>>>(
        g_jt.d_qy_tilde, g_jt.d_h0, g_jt.d_qx0, g_jt.d_qy0, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny, dx, gamma, dt);
    JT_POST_KERNEL();

    JT_CUDA_CHECK(
        cudaMemcpy(g_jt.d_qx_saved, g_jt.d_qx_tilde, nqx * sizeof(float), cudaMemcpyDeviceToDevice));
    transport_qx_advect_k<<<jt_blocks_for(nqxN, threads), threads>>>(
        g_jt.d_qx_tilde, g_jt.d_qx_saved, g_jt.d_h0, g_jt.d_qx0, g_jt.d_qy0, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny,
        dx, halfW, halfD, dt);
    JT_POST_KERNEL();

    JT_CUDA_CHECK(
        cudaMemcpy(g_jt.d_qy_saved, g_jt.d_qy_tilde, nqy * sizeof(float), cudaMemcpyDeviceToDevice));
    transport_qy_advect_k<<<jt_blocks_for(nqyN, threads), threads>>>(
        g_jt.d_qy_tilde, g_jt.d_qy_saved, g_jt.d_h0, g_jt.d_qx0, g_jt.d_qy0, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny,
        dx, halfW, halfD, dt);
    JT_POST_KERNEL();

    transport_h_damp_k<<<jt_blocks_for(ncellN, threads), threads>>>(
        g_jt.d_h_tilde, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny, dx, gamma, dt);
    JT_POST_KERNEL();

    JT_CUDA_CHECK(
        cudaMemcpy(g_jt.d_h_saved, g_jt.d_h_tilde, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    transport_h_advect_k<<<jt_blocks_for(ncellN, threads), threads>>>(
        g_jt.d_h_tilde, g_jt.d_h_saved, g_jt.d_h1, g_jt.d_qx1, g_jt.d_qy1, nx, ny, dx, halfW, halfD, dt);
    JT_POST_KERNEL();
}

} // namespace bp_jt_detail

void transportSurfaceGpu(WaveDecomposition& dec, const Grid& gBar0, const Grid& gBar1, float halfW, float halfD,
                           float dt, float gamma) {
    BP_JT_CUDA_OK(cudaSetDevice(0));
    const int nx = gBar0.NX;
    const int ny = gBar0.NY;
    if (nx != gBar1.NX || ny != gBar1.NY || gBar0.dx != gBar1.dx) {
        std::fprintf(stderr, "transportSurfaceGpu: grid size mismatch\n");
        std::abort();
    }
    const float dx = gBar0.dx;

    bp_jt_detail::g_jt.ensure(nx, ny);
    const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);

    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_h0, gBar0.h.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qx0, gBar0.qx.data(), nqx * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qy0, gBar0.qy.data(), nqy * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_h1, gBar1.h.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qx1, gBar1.qx.data(), nqx * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qy1, gBar1.qy.data(), nqy * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qx_tilde, dec.qx_tilde.data(), nqx * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_qy_tilde, dec.qy_tilde.data(), nqy * sizeof(float), cudaMemcpyHostToDevice));
    BP_JT_CUDA_OK(
        cudaMemcpy(bp_jt_detail::g_jt.d_h_tilde, dec.h_tilde.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));

    bp_jt_detail::runTransportKernels(nx, ny, dx, halfW, halfD, dt, gamma);

    BP_JT_CUDA_OK(cudaDeviceSynchronize());

    BP_JT_CUDA_OK(cudaMemcpy(dec.qx_tilde.data(), bp_jt_detail::g_jt.d_qx_tilde, nqx * sizeof(float),
                             cudaMemcpyDeviceToHost));
    BP_JT_CUDA_OK(cudaMemcpy(dec.qy_tilde.data(), bp_jt_detail::g_jt.d_qy_tilde, nqy * sizeof(float),
                             cudaMemcpyDeviceToHost));
    BP_JT_CUDA_OK(cudaMemcpy(dec.h_tilde.data(), bp_jt_detail::g_jt.d_h_tilde, ncell * sizeof(float),
                             cudaMemcpyDeviceToHost));
}

void transportSurfaceGpuDevice(float* d_bar0_h, float* d_bar0_qx, float* d_bar0_qy, float* d_bar1_h, float* d_bar1_qx,
                               float* d_bar1_qy, float* d_h_tilde, float* d_qx_tilde, float* d_qy_tilde, int nx, int ny,
                               float dx, float halfW, float halfD, float dt, float gamma) {
    BP_JT_CUDA_OK(cudaSetDevice(0));
    bp_jt_detail::g_jt.ensure(nx, ny);
    const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);

    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_h0, d_bar0_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qx0, d_bar0_qx, nqx * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qy0, d_bar0_qy, nqy * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_h1, d_bar1_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qx1, d_bar1_qx, nqx * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qy1, d_bar1_qy, nqy * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qx_tilde, d_qx_tilde, nqx * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_qy_tilde, d_qy_tilde, nqy * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(bp_jt_detail::g_jt.d_h_tilde, d_h_tilde, ncell * sizeof(float), cudaMemcpyDeviceToDevice));

    bp_jt_detail::runTransportKernels(nx, ny, dx, halfW, halfD, dt, gamma);

    BP_JT_CUDA_OK(cudaDeviceSynchronize());

    BP_JT_CUDA_OK(cudaMemcpy(d_qx_tilde, bp_jt_detail::g_jt.d_qx_tilde, nqx * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(d_qy_tilde, bp_jt_detail::g_jt.d_qy_tilde, nqy * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_JT_CUDA_OK(cudaMemcpy(d_h_tilde, bp_jt_detail::g_jt.d_h_tilde, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
}
