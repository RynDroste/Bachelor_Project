#include "solver_pipeline/wave_decompose_gpu.hpp"
#include "solver_pipeline/gpu_terrain_h2d_cache.hpp"

#include "solver_pipeline/wavedecomposer.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>

#include <cuda_runtime.h>

namespace bp_wd_detail {

#define WD_CUDA_CHECK(x)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));               \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

#define WD_POST_KERNEL() WD_CUDA_CHECK(cudaGetLastError())

constexpr float kEps      = 1e-8f;
constexpr float kSigmaMax = 8.f;
constexpr float kStopFlowEps = 0.01f;

__device__ __forceinline__ float harm_d(float a, float b) {
    return 2.f * a * b / (a + b + kEps);
}

__device__ __forceinline__ void atomicMaxFloatNonNeg(float* addr, float val) {
    // Safe for non-negative floats: same bit ordering as int.
    int* int_addr = reinterpret_cast<int*>(addr);
    int old       = *int_addr;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val)
            return;
        old = atomicCAS(int_addr, assumed, __float_as_int(val));
    } while (assumed != old);
}

// Face-local alpha at the right face (alpha_xR) and top face (alpha_yT) of each cell.
// Mirrors Sim2D.cu kernel_init_decomp's face conductance, with:
//   sigma         = max(0, minWaterlevel - maxGround)         (no hard upper clamp)
//   a_main        = tanhf((sigma/sigma_max)^2)                 (soft cap to (0,1))
//   exp(-d_grad * |grad H_total|^2)                            (gradient penalty stays)
//   wet/dry: a = 0 if either side has h <= 0
__global__ void wd_alpha_face_k(const float* __restrict__ d_h,
                                const float* __restrict__ d_b,
                                float* __restrict__ d_alpha_xR,
                                float* __restrict__ d_alpha_yT,
                                int nx,
                                int ny,
                                float d_grad) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i   = idx % nx;
    const int j   = idx / nx;
    const int c   = idx;
    const int ip  = (i + 1 < nx) ? (i + 1) : (nx - 1);
    const int jp  = (j + 1 < ny) ? (j + 1) : (ny - 1);
    const int cxp = ip + j * nx;
    const int cyp = i + jp * nx;

    const float h_c   = d_h[c];
    const float b_c   = d_b[c];
    const float h_xp  = d_h[cxp];
    const float b_xp  = d_b[cxp];
    const float h_yp  = d_h[cyp];
    const float b_yp  = d_b[cyp];

    // Right face: between (i,j) and (i+1,j).
    {
        float a = 0.f;
        if (h_c > 0.f && h_xp > 0.f) {
            const float maxGround = fmaxf(b_c, b_xp);
            const float minWater  = 0.5f * ((b_c + h_c) + (b_xp + h_xp));
            const float sigma     = fmaxf(0.f, minWater - maxGround);
            const float ratio     = sigma / kSigmaMax;
            a                     = tanhf(ratio * ratio);
        }
        const float gradient = fabsf((b_c + h_c) - (b_xp + h_xp));
        d_alpha_xR[c]        = (i + 1 < nx) ? a * expf(-d_grad * gradient * gradient) : 0.f;
    }
    // Top face: between (i,j) and (i,j+1).
    {
        float a = 0.f;
        if (h_c > 0.f && h_yp > 0.f) {
            const float maxGround = fmaxf(b_c, b_yp);
            const float minWater  = 0.5f * ((b_c + h_c) + (b_yp + h_yp));
            const float sigma     = fmaxf(0.f, minWater - maxGround);
            const float ratio     = sigma / kSigmaMax;
            a                     = tanhf(ratio * ratio);
        }
        const float gradient = fabsf((b_c + h_c) - (b_yp + h_yp));
        d_alpha_yT[c]        = (j + 1 < ny) ? a * expf(-d_grad * gradient * gradient) : 0.f;
    }
}

// alpha_cell[i,j] = min over the 4 face-local alphas at that cell.
// Boundary (i=0 / j=0): missing left/bottom face -> use the existing right/top alpha as a fallback.
__global__ void wd_alpha_cell_k(const float* __restrict__ d_alpha_xR,
                                const float* __restrict__ d_alpha_yT,
                                float* __restrict__ d_alpha_cell,
                                int nx,
                                int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;

    const float aR = d_alpha_xR[idx];
    const float aT = d_alpha_yT[idx];
    const float aL = (i > 0) ? d_alpha_xR[(i - 1) + j * nx] : aR;
    const float aB = (j > 0) ? d_alpha_yT[i + (j - 1) * nx] : aT;
    d_alpha_cell[idx] = fminf(fminf(aR, aL), fminf(aT, aB));
}

// Block-stride max reduction with atomic into a single device float.
__global__ void wd_alpha_max_k(const float* __restrict__ a, float* __restrict__ out, int n) {
    extern __shared__ float sdata[];
    const int tid    = threadIdx.x;
    const int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    float     local  = 0.f;
    for (int k = gid; k < n; k += stride)
        local = fmaxf(local, a[k]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMaxFloatNonNeg(out, sdata[0]);
}

// Variable-coefficient 5-point Laplacian. Face conductance is the harmonic mean
// of the two adjacent cell-centered alphas. Boundary faces have zero conductance
// (zero-flux Neumann). The whole domain shares the same dt_global passed in.
__global__ void wd_diffuse_iter_k(const float* __restrict__ u,
                                  float* __restrict__ u_new,
                                  const float* __restrict__ alpha_cell,
                                  int nx,
                                  int ny,
                                  float dx,
                                  float dt_global) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;

    const float aC = alpha_cell[idx];
    const float aR = (i + 1 < nx) ? harm_d(aC, alpha_cell[(i + 1) + j * nx]) : 0.f;
    const float aL = (i > 0)      ? harm_d(alpha_cell[(i - 1) + j * nx], aC) : 0.f;
    const float aT = (j + 1 < ny) ? harm_d(aC, alpha_cell[i + (j + 1) * nx]) : 0.f;
    const float aB = (j > 0)      ? harm_d(alpha_cell[i + (j - 1) * nx], aC) : 0.f;

    const float uC = u[idx];
    const float uR = (i + 1 < nx) ? u[(i + 1) + j * nx] : uC;
    const float uL = (i > 0)      ? u[(i - 1) + j * nx] : uC;
    const float uT = (j + 1 < ny) ? u[i + (j + 1) * nx] : uC;
    const float uB = (j > 0)      ? u[i + (j - 1) * nx] : uC;

    const float lap = aR * (uR - uC) - aL * (uC - uL) + aT * (uT - uC) - aB * (uC - uB);
    u_new[idx]      = uC + dt_global * (lap / (dx * dx));
}

// Convert staggered face-array qx[i + j*(nx+1)] (size (nx+1)*ny) to a cell-centered
// "right-face-of-cell" buffer: qxc[i,j] = qx[(i+1) + j*(nx+1)].
__global__ void wd_face_to_qxc_k(const float* __restrict__ qx_face,
                                 float* __restrict__ qxc,
                                 int nx,
                                 int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    qxc[idx]    = qx_face[(i + 1) + j * (nx + 1)];
}

__global__ void wd_face_to_qyc_k(const float* __restrict__ qy_face,
                                 float* __restrict__ qyc,
                                 int nx,
                                 int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    qyc[idx]    = qy_face[i + (j + 1) * nx];
}

__global__ void wd_add_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = a[idx] + b[idx];
}

// 2D StopFlow predicates mirroring Sim2D.cu's StopFlowOnTerrainBoundary_{x,y}_d.
__device__ __forceinline__ bool wd_stop_flow_x(int i, int j, const float* h, const float* b, int nx) {
    const int ip1 = (i + 1 < nx) ? (i + 1) : (nx - 1);
    const int c   = i + j * nx;
    const int cxp = ip1 + j * nx;
    if ((h[c]   <= kStopFlowEps) && (b[c]   >= b[cxp] + h[cxp])) return true;
    if ((h[cxp] <= kStopFlowEps) && (b[cxp] >  b[c]   + h[c]))   return true;
    return false;
}

__device__ __forceinline__ bool wd_stop_flow_y(int i, int j, const float* h, const float* b, int nx, int ny) {
    const int jp1 = (j + 1 < ny) ? (j + 1) : (ny - 1);
    const int c   = i + j * nx;
    const int cyp = i + jp1 * nx;
    if ((h[c]   <= kStopFlowEps) && (b[c]   >= b[cyp] + h[cyp])) return true;
    if ((h[cyp] <= kStopFlowEps) && (b[cyp] >  b[c]   + h[c]))   return true;
    return false;
}

// Final split: write h_bar/h_tilde and qx_bar/qx_tilde/qy_bar/qy_tilde, then apply
// StopFlow to zero out momentum on dry/terrain boundaries.
__global__ void wd_final_split_k(const float* __restrict__ H_bar,
                                 const float* __restrict__ d_h,
                                 const float* __restrict__ d_b,
                                 const float* qxc_bar,
                                 const float* qyc_bar,
                                 const float* __restrict__ qxc_in,
                                 const float* __restrict__ qyc_in,
                                 float* __restrict__ out_h_bar,
                                 float* __restrict__ out_h_tilde,
                                 float* out_qx_bar,
                                 float* __restrict__ out_qx_tilde,
                                 float* out_qy_bar,
                                 float* __restrict__ out_qy_tilde,
                                 int nx,
                                 int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;

    const float hb = fmaxf(0.f, H_bar[idx] - d_b[idx]);
    out_h_bar[idx]   = hb;
    out_h_tilde[idx] = d_h[idx] - hb;

    float qxb = qxc_bar[idx];
    float qxt = qxc_in[idx] - qxb;
    float qyb = qyc_bar[idx];
    float qyt = qyc_in[idx] - qyb;

    if (wd_stop_flow_x(i, j, d_h, d_b, nx)) {
        qxb = 0.f;
        qxt = 0.f;
    }
    if (wd_stop_flow_y(i, j, d_h, d_b, nx, ny)) {
        qyb = 0.f;
        qyt = 0.f;
    }
    out_qx_bar[idx]   = qxb;
    out_qx_tilde[idx] = qxt;
    out_qy_bar[idx]   = qyb;
    out_qy_tilde[idx] = qyt;
}

inline int blocks_for(int n, int threads = 256) {
    return (n + threads - 1) / threads;
}

void wd_run_diffuse(const float* d_src,
                    float* d_dst,
                    float* d_a,
                    float* d_b,
                    const float* alpha_cell,
                    int nx,
                    int ny,
                    float dx,
                    float dt_global,
                    int nIter) {
    const int     n       = nx * ny;
    const size_t  bytes   = static_cast<size_t>(n) * sizeof(float);
    constexpr int threads = 256;
    const int     blks    = blocks_for(n, threads);
    if (nIter <= 0) {
        if (d_dst != d_src)
            WD_CUDA_CHECK(cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice));
        return;
    }
    if (d_a != d_src)
        WD_CUDA_CHECK(cudaMemcpy(d_a, d_src, bytes, cudaMemcpyDeviceToDevice));
    float* in  = d_a;
    float* out = d_b;
    for (int it = 0; it < nIter; ++it) {
        wd_diffuse_iter_k<<<blks, threads, 0, 0>>>(in, out, alpha_cell, nx, ny, dx, dt_global);
        WD_POST_KERNEL();
        float* t = in;
        in       = out;
        out      = t;
    }
    if (in != d_dst)
        WD_CUDA_CHECK(cudaMemcpy(d_dst, in, bytes, cudaMemcpyDeviceToDevice));
}

struct WdGpuScratch {
    int nx = 0;
    int ny = 0;

    float* d_h          = nullptr;
    float* d_b          = nullptr;
    float* d_alpha_xR   = nullptr;
    float* d_alpha_yT   = nullptr;
    float* d_alpha_cell = nullptr;
    float* d_alpha_max  = nullptr; // 1-float reduction output

    float* d_ping       = nullptr; // diffusion ping/pong (ncell)
    float* d_pong       = nullptr;

    float* d_qxc_in     = nullptr; // input cell-centered qx (right-face-of-cell)
    float* d_qyc_in     = nullptr; // input cell-centered qy (top-face-of-cell)
    float* d_qx_face_in = nullptr; // staggered upload buffer for qx
    float* d_qy_face_in = nullptr; // staggered upload buffer for qy

    float* d_h_bar      = nullptr;
    float* d_h_tilde    = nullptr;
    float* d_qx_bar     = nullptr;
    float* d_qx_tilde   = nullptr;
    float* d_qy_bar     = nullptr;
    float* d_qy_tilde   = nullptr;

    void freeAll() {
        cudaFree(d_h);
        cudaFree(d_b);
        cudaFree(d_alpha_xR);
        cudaFree(d_alpha_yT);
        cudaFree(d_alpha_cell);
        cudaFree(d_alpha_max);
        cudaFree(d_ping);
        cudaFree(d_pong);
        cudaFree(d_qxc_in);
        cudaFree(d_qyc_in);
        cudaFree(d_qx_face_in);
        cudaFree(d_qy_face_in);
        cudaFree(d_h_bar);
        cudaFree(d_h_tilde);
        cudaFree(d_qx_bar);
        cudaFree(d_qx_tilde);
        cudaFree(d_qy_bar);
        cudaFree(d_qy_tilde);
        d_h = d_b = d_alpha_xR = d_alpha_yT = d_alpha_cell = d_alpha_max = nullptr;
        d_ping = d_pong = nullptr;
        d_qxc_in = d_qyc_in = d_qx_face_in = d_qy_face_in = nullptr;
        d_h_bar = d_h_tilde = d_qx_bar = d_qx_tilde = d_qy_bar = d_qy_tilde = nullptr;
        nx = ny = 0;
        bp_gpu::terrainCacheInvalidate();
    }

    void ensure(int nx_, int ny_) {
        if (nx_ == nx && ny_ == ny)
            return;
        freeAll();
        nx = nx_;
        ny = ny_;
        const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
        const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
        const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);
        const size_t bF    = ncell * sizeof(float);
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_xR), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_yT), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_cell), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_alpha_max), sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ping), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pong), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qxc_in), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qyc_in), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_face_in), nqx * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_face_in), nqy * sizeof(float)));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_bar), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_tilde), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_bar), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_tilde), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_bar), bF));
        WD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_tilde), bF));
    }
};

WdGpuScratch g_wd;

float compute_dt_global(float* d_alpha_cell, float* d_alpha_max, int ncell, float dx) {
    constexpr int threads = 256;
    constexpr int blocks  = 64; // bounded; each thread handles a stride
    WD_CUDA_CHECK(cudaMemsetAsync(d_alpha_max, 0, sizeof(float), 0));
    wd_alpha_max_k<<<blocks, threads, threads * sizeof(float), 0>>>(d_alpha_cell, d_alpha_max, ncell);
    WD_POST_KERNEL();
    float a_max_host = 0.f;
    WD_CUDA_CHECK(cudaMemcpy(&a_max_host, d_alpha_max, sizeof(float), cudaMemcpyDeviceToHost));
    const float dt_global = fminf(0.25f, dx * dx / (4.f * a_max_host + kEps));
    return dt_global;
}

void wdRunDecomposeCore(const Grid& g, float d_grad_penalty, int n_diffusion_iters) {
    const int   nx    = g.NX;
    const int   ny    = g.NY;
    const int   nIter = std::max(0, n_diffusion_iters);
    const float dx    = g.dx;

    g_wd.ensure(nx, ny);
    const int ncell = nx * ny;
    const int nqx   = (nx + 1) * ny;
    const int nqy   = nx * (ny + 1);

    constexpr int threads = 256;
    const int     blks    = blocks_for(ncell, threads);

    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_h, g.h.data(), static_cast<size_t>(ncell) * sizeof(float), cudaMemcpyHostToDevice));
    if (!bp_gpu::terrainHostMatchesCachedSnapshot(g.terrain.data(), static_cast<std::size_t>(ncell))) {
        WD_CUDA_CHECK(cudaMemcpy(g_wd.d_b, g.terrain.data(), static_cast<size_t>(ncell) * sizeof(float),
                                 cudaMemcpyHostToDevice));
        bp_gpu::noteWaveDecomposeTerrainH2d(g.terrain.data(), static_cast<std::size_t>(ncell));
    }

    // Upload staggered qx/qy and convert to cell-centered (right/top-face-of-cell) buffers.
    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_qx_face_in, g.qx.data(), static_cast<size_t>(nqx) * sizeof(float),
                             cudaMemcpyHostToDevice));
    WD_CUDA_CHECK(cudaMemcpy(g_wd.d_qy_face_in, g.qy.data(), static_cast<size_t>(nqy) * sizeof(float),
                             cudaMemcpyHostToDevice));
    wd_face_to_qxc_k<<<blks, threads, 0, 0>>>(g_wd.d_qx_face_in, g_wd.d_qxc_in, nx, ny);
    WD_POST_KERNEL();
    wd_face_to_qyc_k<<<blks, threads, 0, 0>>>(g_wd.d_qy_face_in, g_wd.d_qyc_in, nx, ny);
    WD_POST_KERNEL();

    // Face-local alpha, then cell-min alpha, then global dt.
    wd_alpha_face_k<<<blks, threads, 0, 0>>>(g_wd.d_h, g_wd.d_b, g_wd.d_alpha_xR, g_wd.d_alpha_yT, nx, ny,
                                             d_grad_penalty);
    WD_POST_KERNEL();
    wd_alpha_cell_k<<<blks, threads, 0, 0>>>(g_wd.d_alpha_xR, g_wd.d_alpha_yT, g_wd.d_alpha_cell, nx, ny);
    WD_POST_KERNEL();
    const float dt_global = compute_dt_global(g_wd.d_alpha_cell, g_wd.d_alpha_max, ncell, dx);

    // Diffuse qxc and qyc first, since they only need scratch buffers (d_ping/d_pong)
    // and write into dedicated output buffers (d_qx_bar / d_qy_bar).
    wd_run_diffuse(g_wd.d_qxc_in, g_wd.d_qx_bar, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha_cell, nx, ny, dx, dt_global,
                   nIter);
    wd_run_diffuse(g_wd.d_qyc_in, g_wd.d_qy_bar, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha_cell, nx, ny, dx, dt_global,
                   nIter);

    // Diffuse the total water surface H = h + b LAST, leaving the smoothed result
    // in d_ping (so it survives until wd_final_split_k consumes it).
    wd_add_k<<<blks, threads, 0, 0>>>(g_wd.d_h, g_wd.d_b, g_wd.d_ping, ncell);
    WD_POST_KERNEL();
    wd_run_diffuse(g_wd.d_ping, g_wd.d_ping, g_wd.d_ping, g_wd.d_pong, g_wd.d_alpha_cell, nx, ny, dx, dt_global, nIter);

    // Final split: writes h_bar/h_tilde, and re-derives qx_bar/qx_tilde/qy_bar/qy_tilde
    // (the bar fields above get overwritten with the StopFlow-corrected values).
    wd_final_split_k<<<blks, threads, 0, 0>>>(g_wd.d_ping, g_wd.d_h, g_wd.d_b, g_wd.d_qx_bar, g_wd.d_qy_bar,
                                              g_wd.d_qxc_in, g_wd.d_qyc_in, g_wd.d_h_bar, g_wd.d_h_tilde, g_wd.d_qx_bar,
                                              g_wd.d_qx_tilde, g_wd.d_qy_bar, g_wd.d_qy_tilde, nx, ny);
    WD_POST_KERNEL();
}

} // namespace bp_wd_detail

void waveDecomposeGpu(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out) {
    WD_CUDA_CHECK(cudaSetDevice(0));
    bp_wd_detail::wdRunDecomposeCore(g, d_grad_penalty, n_diffusion_iters);
    const int   nx    = g.NX;
    const int   ny    = g.NY;
    const int   ncell = nx * ny;
    auto&       g_wd  = bp_wd_detail::g_wd;
    const size_t bF   = static_cast<size_t>(ncell) * sizeof(float);

    out.h_bar.resize(static_cast<size_t>(ncell));
    out.h_tilde.resize(static_cast<size_t>(ncell));
    out.qx_bar.resize(static_cast<size_t>(ncell));
    out.qx_tilde.resize(static_cast<size_t>(ncell));
    out.qy_bar.resize(static_cast<size_t>(ncell));
    out.qy_tilde.resize(static_cast<size_t>(ncell));

    WD_CUDA_CHECK(cudaMemcpy(out.h_bar.data(),    g_wd.d_h_bar,    bF, cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.h_tilde.data(),  g_wd.d_h_tilde,  bF, cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qx_bar.data(),   g_wd.d_qx_bar,   bF, cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qx_tilde.data(), g_wd.d_qx_tilde, bF, cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qy_bar.data(),   g_wd.d_qy_bar,   bF, cudaMemcpyDeviceToHost));
    WD_CUDA_CHECK(cudaMemcpy(out.qy_tilde.data(), g_wd.d_qy_tilde, bF, cudaMemcpyDeviceToHost));
}

WaveDecompGpuPtrs waveDecomposeGpuDeviceOnly(const Grid& g, float d_grad_penalty, int n_diffusion_iters) {
    WD_CUDA_CHECK(cudaSetDevice(0));
    bp_wd_detail::wdRunDecomposeCore(g, d_grad_penalty, n_diffusion_iters);
    auto& w = bp_wd_detail::g_wd;
    WaveDecompGpuPtrs p{};
    p.nx            = g.NX;
    p.ny            = g.NY;
    p.d_h_bar       = w.d_h_bar;
    p.d_h_tilde     = w.d_h_tilde;
    p.d_qx_bar      = w.d_qx_bar;
    p.d_qx_tilde    = w.d_qx_tilde;
    p.d_qy_bar      = w.d_qy_bar;
    p.d_qy_tilde    = w.d_qy_tilde;
    return p;
}
