// GPU wave decomposition (Jeschke & Wojtan-style low-pass / tilde split).

#include "wave_decompose_gpu.hpp"

#include "wavedecomposer.h"

#include <algorithm>
#include <cstdio>

#include <cuda_runtime.h>

namespace {

#define WD_CUDA_CHECK(x)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define WD_POST_KERNEL() WD_CUDA_CHECK(cudaGetLastError())

constexpr float kEps = 1e-8f;

__device__ __forceinline__ float harm_d(float a, float b) {
    return 2.f * a * b / (a + b + kEps);
}

__global__ void wd_alpha_from_h_k(const float* __restrict__ d_h,
                                  float* __restrict__ d_alpha,
                                  int nx,
                                  int ny,
                                  float dx,
                                  float d_grad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    const int im = max(0, i - 1);
    const int ip = min(nx - 1, i + 1);
    const int jm = max(0, j - 1);
    const int jp = min(ny - 1, j + 1);
    const float denom_x = static_cast<float>(ip - im) * dx;
    const float denom_y = static_cast<float>(jp - jm) * dx;
    const float grad_h_x =
        (denom_x > 0.f) ? (d_h[ip + j * nx] - d_h[im + j * nx]) / denom_x : 0.f;
    const float grad_h_y =
        (denom_y > 0.f) ? (d_h[i + jp * nx] - d_h[i + jm * nx]) / denom_y : 0.f;
    const float grad_h_sq = grad_h_x * grad_h_x + grad_h_y * grad_h_y;
    const float h         = d_h[idx];
    d_alpha[idx]          = (h * h / 64.f) * expf(-d_grad * grad_h_sq);
}

__global__ void wd_add_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = a[idx] + b[idx];
}

__global__ void wd_diffuse_iter_k(const float* __restrict__ u,
                                  float* __restrict__ u_new,
                                  const float* __restrict__ alpha,
                                  int nx,
                                  int ny,
                                  float dx) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;

    const float aij = alpha[idx];
    const int ip = min(i + 1, nx - 1);
    const int im = max(i - 1, 0);
    const int jp = min(j + 1, ny - 1);
    const int jm = max(j - 1, 0);

    const float aR = (i + 1 < nx) ? harm_d(aij, alpha[ip + j * nx]) : 0.f;
    const float aL = (i - 1 >= 0) ? harm_d(alpha[im + j * nx], aij) : 0.f;
    const float aT = (j + 1 < ny) ? harm_d(aij, alpha[i + jp * nx]) : 0.f;
    const float aB = (j - 1 >= 0) ? harm_d(alpha[i + jm * nx], aij) : 0.f;

    const float uC = u[idx];
    const float uR = (i + 1 < nx) ? u[ip + j * nx] : uC;
    const float uL = (i - 1 >= 0) ? u[im + j * nx] : uC;
    const float uT = (j + 1 < ny) ? u[i + jp * nx] : uC;
    const float uB = (j - 1 >= 0) ? u[i + jm * nx] : uC;

    float a_max = fmaxf(fmaxf(aR, aL), fmaxf(aT, aB));
    a_max       = fmaxf(a_max, kEps);
    const float dx2    = dx * dx;
    const float dt_loc = fminf(0.25f, dx2 / (4.f * a_max + kEps));

    float lap = aR * (uR - uC) - aL * (uC - uL) + aT * (uT - uC) - aB * (uC - uB);
    lap /= dx2;
    u_new[idx] = uC + dt_loc * lap;
}

__global__ void wd_h_split_k(const float* __restrict__ H_bar,
                             const float* __restrict__ d_b,
                             const float* __restrict__ d_h,
                             float* __restrict__ out_h_bar,
                             float* __restrict__ out_h_tilde,
                             int nx,
                             int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const float hb = fmaxf(0.f, H_bar[idx] - d_b[idx]);
    out_h_bar[idx]   = hb;
    out_h_tilde[idx] = d_h[idx] - hb;
}

__global__ void wd_alpha_qx_k(const float* __restrict__ alpha_cell, float* __restrict__ alpha_qx, int nx, int ny) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = (nx + 1) * ny;
    if (tid >= n)
        return;
    const int i = tid % (nx + 1);
    const int j = tid / (nx + 1);
    float a;
    if (i == 0)
        a = alpha_cell[0 + j * nx];
    else if (i == nx)
        a = alpha_cell[(nx - 1) + j * nx];
    else
        a = harm_d(alpha_cell[(i - 1) + j * nx], alpha_cell[i + j * nx]);
    alpha_qx[tid] = a;
}

__global__ void wd_alpha_qy_k(const float* __restrict__ alpha_cell, float* __restrict__ alpha_qy, int nx, int ny) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * (ny + 1);
    if (tid >= n)
        return;
    const int i      = tid % nx;
    const int j_face = tid / nx;
    float a;
    if (j_face == 0)
        a = alpha_cell[i + 0 * nx];
    else if (j_face == ny)
        a = alpha_cell[i + (ny - 1) * nx];
    else
        a = harm_d(alpha_cell[i + (j_face - 1) * nx], alpha_cell[i + j_face * nx]);
    alpha_qy[tid] = a;
}

__global__ void wd_vec_sub_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = a[idx] - b[idx];
}

int blocks_for(int n, int threads = 256) {
    return (n + threads - 1) / threads;
}

// Ping-pong diffuse; d_a and d_b must differ. Final field ends in d_dst (copy if needed).
void wd_run_diffuse(const float* d_src,
                    float* d_dst,
                    float* d_a,
                    float* d_b,
                    const float* alpha,
                    int w,
                    int h,
                    float dx,
                    int nIter) {
    const int         n       = w * h;
    constexpr int     threads = 256;
    const int         blks    = blocks_for(n, threads);
    if (nIter <= 0) {
        WD_CUDA_CHECK(cudaMemcpy(d_dst, d_src, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }
    WD_CUDA_CHECK(cudaMemcpy(d_a, d_src, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice));
    float* in  = d_a;
    float* out = d_b;
    for (int it = 0; it < nIter; ++it) {
        wd_diffuse_iter_k<<<blks, threads, 0, 0>>>(in, out, alpha, w, h, dx);
        WD_POST_KERNEL();
        float* t = in;
        in       = out;
        out      = t;
    }
    if (in != d_dst)
        WD_CUDA_CHECK(cudaMemcpy(d_dst, in, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice));
}

struct WdGpuScratch {
    int nx = 0;
    int ny = 0;

    float* d_h        = nullptr;
    float* d_b        = nullptr;
    float* d_alpha    = nullptr;
    float* d_alpha_qx = nullptr;
    float* d_alpha_qy = nullptr;
    float* d_ping     = nullptr;
    float* d_pong     = nullptr;
    float* d_qx_src   = nullptr;
    float* d_qy_src   = nullptr;
    float* d_h_bar    = nullptr;
    float* d_h_tilde  = nullptr;
    float* d_qx_bar   = nullptr;
    float* d_qx_tilde = nullptr;
    float* d_qy_bar   = nullptr;
    float* d_qy_tilde = nullptr;

    void freeAll() {
        cudaFree(d_h);
        cudaFree(d_b);
        cudaFree(d_alpha);
        cudaFree(d_alpha_qx);
        cudaFree(d_alpha_qy);
        cudaFree(d_ping);
        cudaFree(d_pong);
        cudaFree(d_qx_src);
        cudaFree(d_qy_src);
        cudaFree(d_h_bar);
        cudaFree(d_h_tilde);
        cudaFree(d_qx_bar);
        cudaFree(d_qx_tilde);
        cudaFree(d_qy_bar);
        cudaFree(d_qy_tilde);
        d_h = d_b = d_alpha = d_alpha_qx = d_alpha_qy = nullptr;
        d_ping = d_pong = d_qx_src = d_qy_src = nullptr;
        d_h_bar = d_h_tilde = d_qx_bar = d_qx_tilde = d_qy_bar = d_qy_tilde = nullptr;
        nx = ny = 0;
    }

    void ensure(int nx_, int ny_) {
        if (nx_ == nx && ny_ == ny)
            return;
        freeAll();
        nx = nx_;
        ny = ny_;
        const int ncell = nx * ny;
        const int nqx   = (nx + 1) * ny;
        const int nqy   = nx * (ny + 1);
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h), static_cast<size_t>(ncell) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), static_cast<size_t>(ncell) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha), static_cast<size_t>(ncell) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_qx), static_cast<size_t>(nqx) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_qy), static_cast<size_t>(nqy) * sizeof(float)));
        const size_t mx = static_cast<size_t>(std::max({ncell, nqx, nqy}));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ping), mx * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pong), mx * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_src), static_cast<size_t>(nqx) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_src), static_cast<size_t>(nqy) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_bar), static_cast<size_t>(ncell) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_tilde), static_cast<size_t>(ncell) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_bar), static_cast<size_t>(nqx) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_tilde), static_cast<size_t>(nqx) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_bar), static_cast<size_t>(nqy) * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_tilde), static_cast<size_t>(nqy) * sizeof(float)));
    }
};

WdGpuScratch g_wd;

} // namespace

void waveDecomposeGpu(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out) {
    WD_CUDA_CHECK(cudaSetDevice(0));
    const int nx    = g.NX;
    const int ny    = g.NY;
    const int nIter = std::max(0, n_diffusion_iters);
    const float dx  = g.dx;

    g_wd.ensure(nx, ny);
    const int ncell = nx * ny;
    const int nqx   = (nx + 1) * ny;
    const int nqy   = nx * (ny + 1);

    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_h, g.h.data(), static_cast<size_t>(ncell) * sizeof(float), cudaMemcpyHostToDevice));
    WD_CUDA_CHECK(
        cudaMemcpy(g_wd.d_b, g.terrain.data(), static_cast<size_t>(ncell) * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    wd_alpha_from_h_k<<<blocks_for(ncell, threads), threads, 0, 0>>>(
        g_wd.d_h, g_wd.d_alpha, nx, ny, dx, d_grad_penalty);
    WD_POST_KERNEL();

    wd_add_k<<<blocks_for(ncell, threads), threads, 0, 0>>>(g_wd.d_h, g_wd.d_b, g_wd.d_ping, ncell);
    WD_POST_KERNEL();

    wd_run_diffuse(g_wd.d_ping, g_wd.d_ping, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha, nx, ny, dx, nIter);

    wd_h_split_k<<<blocks_for(ncell, threads), threads, 0, 0>>>(
        g_wd.d_ping, g_wd.d_b, g_wd.d_h, g_wd.d_h_bar, g_wd.d_h_tilde, nx, ny);
    WD_POST_KERNEL();

    wd_alpha_qx_k<<<blocks_for(nqx, threads), threads, 0, 0>>>(g_wd.d_alpha, g_wd.d_alpha_qx, nx, ny);
    WD_POST_KERNEL();
    wd_alpha_qy_k<<<blocks_for(nqy, threads), threads, 0, 0>>>(g_wd.d_alpha, g_wd.d_alpha_qy, nx, ny);
    WD_POST_KERNEL();

    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_qx_src, g.qx.data(), static_cast<size_t>(nqx) * sizeof(float), cudaMemcpyHostToDevice));
    wd_run_diffuse(g_wd.d_qx_src, g_wd.d_qx_bar, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha_qx, nx + 1, ny, dx, nIter);

    wd_vec_sub_k<<<blocks_for(nqx, threads), threads, 0, 0>>>(
        g_wd.d_qx_src, g_wd.d_qx_bar, g_wd.d_qx_tilde, nqx);
    WD_POST_KERNEL();

    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_qy_src, g.qy.data(), static_cast<size_t>(nqy) * sizeof(float), cudaMemcpyHostToDevice));
    wd_run_diffuse(g_wd.d_qy_src, g_wd.d_qy_bar, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha_qy, nx, ny + 1, dx, nIter);

    wd_vec_sub_k<<<blocks_for(nqy, threads), threads, 0, 0>>>(
        g_wd.d_qy_src, g_wd.d_qy_bar, g_wd.d_qy_tilde, nqy);
    WD_POST_KERNEL();

    out.h_bar.resize(static_cast<size_t>(ncell));
    out.h_tilde.resize(static_cast<size_t>(ncell));
    out.qx_bar.resize(static_cast<size_t>(nqx));
    out.qx_tilde.resize(static_cast<size_t>(nqx));
    out.qy_bar.resize(static_cast<size_t>(nqy));
    out.qy_tilde.resize(static_cast<size_t>(nqy));

    WD_CUDA_CHECK(cudaMemcpy(out.h_bar.data(), g_wd.d_h_bar, static_cast<size_t>(ncell) * sizeof(float),
                             cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.h_tilde.data(), g_wd.d_h_tilde, static_cast<size_t>(ncell) * sizeof(float),
                             cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qx_bar.data(), g_wd.d_qx_bar, static_cast<size_t>(nqx) * sizeof(float),
                             cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qx_tilde.data(), g_wd.d_qx_tilde, static_cast<size_t>(nqx) * sizeof(float),
                             cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qy_bar.data(), g_wd.d_qy_bar, static_cast<size_t>(nqy) * sizeof(float),
                             cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qy_tilde.data(), g_wd.d_qy_tilde, static_cast<size_t>(nqy) * sizeof(float),
                             cudaMemcpyDeviceToHost));
}
