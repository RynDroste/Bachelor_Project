#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/gpu_terrain_h2d_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define BP_SWE_CUDA_OK(x)                                                                                              \
    do {                                                                                                               \
        cudaError_t _e = (x);                                                                                          \
        if (_e != cudaSuccess) {                                                                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace bp_swe_detail {

#define SWE_CUDA_CHECK(x)                                                                                               \
    do {                                                                                                                \
        cudaError_t _e = (x);                                                                                           \
        if (_e != cudaSuccess) {                                                                                        \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));              \
            std::abort();                                                                                               \
        }                                                                                                               \
    } while (0)

#define SWE_POST_KERNEL() SWE_CUDA_CHECK(cudaGetLastError())

constexpr float kG         = 9.81f;
constexpr float kDryEps    = 1e-3f;
constexpr float kCflFactor = 75.f;
constexpr float kCflWave   = 0.8f;

// All buffers (h, b, qx, qy) are now N*N cell-centered. qx[c] is the momentum
// at the RIGHT face of cell (i,j) (between cells (i,j) and (i+1,j)); qy[c] is
// the momentum at the TOP face of cell (i,j). The right face of cell (NX-1, j)
// is the right wall (forced to 0); the left face of cell (0, j) is the left
// wall and is not stored. Symmetric for top/bottom in y.
__device__ __forceinline__ int idxH(int i, int j, int nx) { return i + j * nx; }

__device__ __forceinline__ int clamp_x_d(int i, int nx) {
    return i < 0 ? 0 : (i >= nx ? nx - 1 : i);
}
__device__ __forceinline__ int clamp_y_d(int j, int ny) {
    return j < 0 ? 0 : (j >= ny ? ny - 1 : j);
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float faceSpeedCap_d(float hf, float dx, float dt) {
    const float u_adv  = dx / (kCflFactor * dt);
    const float c      = sqrtf(fmaxf(0.f, kG * hf));
    const float u_wave = kCflWave * dx / dt - c;
    return fminf(u_adv, fmaxf(u_wave, 0.f));
}

// Sign-upwinded velocity at the right face of cell (i,j): u = qx[c] / h_upwind,
// where h_upwind picks h[c] on positive flow and h[cell-to-the-right] on negative
// flow. Mirrors Sim2D's kernel_qbar_to_ubar_x semantics with replicate clamping.
__device__ __forceinline__ float u_at_face_x_d(const float* qx, const float* h, int nx, int i, int j) {
    const int   c   = idxH(i, j, nx);
    const int   cxp = idxH(clamp_x_d(i + 1, nx), j, nx);
    const float q   = qx[c];
    return (q >= 0.f) ? q / fmaxf(kDryEps, h[c]) : q / fmaxf(kDryEps, h[cxp]);
}

__device__ __forceinline__ float u_at_face_y_d(const float* qy, const float* h, int nx, int ny, int i, int j) {
    const int   c   = idxH(i, j, nx);
    const int   cyp = idxH(i, clamp_y_d(j + 1, ny), nx);
    const float q   = qy[c];
    return (q >= 0.f) ? q / fmaxf(kDryEps, h[c]) : q / fmaxf(kDryEps, h[cyp]);
}

__global__ void bc_domain_qx_k(float* qx, int nx, int ny) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny)
        return;
    qx[idxH(nx - 1, j, nx)] = 0.f;
}

__global__ void bc_domain_qy_k(float* qy, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx)
        return;
    qy[idxH(i, ny - 1, nx)] = 0.f;
}

__global__ void bc_terrain_qx_k(float* qx, const float* h, const float* terrain, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx - 1 || j >= ny)
        return;
    const int   c    = idxH(i, j, nx);
    const int   cxp  = idxH(i + 1, j, nx);
    const float bmax = fmaxf(terrain[c], terrain[cxp]);
    const float wL   = terrain[c]   + h[c];
    const float wR   = terrain[cxp] + h[cxp];
    if (bmax >= fminf(wL, wR) - kDryEps)
        qx[c] = 0.f;
}

__global__ void bc_terrain_qy_k(float* qy, const float* h, const float* terrain, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny - 1)
        return;
    const int   c    = idxH(i, j, nx);
    const int   cyp  = idxH(i, j + 1, nx);
    const float bmax = fmaxf(terrain[c], terrain[cyp]);
    const float wD   = terrain[c]   + h[c];
    const float wU   = terrain[cyp] + h[cyp];
    if (bmax >= fminf(wD, wU) - kDryEps)
        qy[c] = 0.f;
}

// Stelling-Duinmeijer momentum update (Sim2D.cu kernel_swe_momentum_x), ported
// to our cell-centered q convention. The control volume for qx[c] is centred
// on the right face of cell (i,j) and spans cells (i,j) and (i+1,j) in x and
// rows j-1, j, j+1 in y. Closed-wall replicate via clamp_x_d / clamp_y_d.
__global__ void step_qx_k(const float* h, const float* b, const float* qx, const float* qy, float* qx_out, int nx,
                          int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;

    const int c = idxH(i, j, nx);
    qx_out[c]   = qx[c];

    if (i >= nx - 1) {
        qx_out[c] = 0.f;
        return;
    }

    const int cxp = idxH(i + 1, j, nx);

    if (h[c] + h[cxp] < 2.f * kDryEps) {
        qx_out[c] = 0.f;
        return;
    }

    const int im1 = clamp_x_d(i - 1, nx);
    const int ip1 = clamp_x_d(i + 1, nx);
    const int ip2 = clamp_x_d(i + 2, nx);
    const int jm1 = clamp_y_d(j - 1, ny);
    const int jp1 = clamp_y_d(j + 1, ny);

    const int cxm    = idxH(im1, j,   nx);
    const int cxpp   = idxH(ip2, j,   nx);
    const int cym    = idxH(i,   jm1, nx);
    const int cyp    = idxH(i,   jp1, nx);
    const int cxp_ym = idxH(ip1, jm1, nx);
    const int cxp_yp = idxH(ip1, jp1, nx);

    const float u_self = u_at_face_x_d(qx, h, nx, i,   j);
    const float u_xL   = u_at_face_x_d(qx, h, nx, im1, j);
    const float u_xR   = u_at_face_x_d(qx, h, nx, ip1, j);
    const float u_yB   = u_at_face_x_d(qx, h, nx, i,   jm1);
    const float u_yT   = u_at_face_x_d(qx, h, nx, i,   jp1);

    const float qfx_L = (u_xL   >= 0.f) ? u_xL   * h[cxm] : u_xL   * h[c];
    const float qfx_S = (u_self >= 0.f) ? u_self * h[c]   : u_self * h[cxp];
    const float qfx_R = (u_xR   >= 0.f) ? u_xR   * h[cxp] : u_xR   * h[cxpp];

    const float qBarX_L  = 0.5f * (qfx_L + qfx_S);
    const float qBarX_R  = 0.5f * (qfx_S + qfx_R);
    const float uStarX_L = (qBarX_L >= 0.f) ? u_xL   : u_self;
    const float uStarX_R = (qBarX_R >  0.f) ? u_self : u_xR;

    const float v_BL = u_at_face_y_d(qy, h, nx, ny, i,   jm1);
    const float v_BR = u_at_face_y_d(qy, h, nx, ny, ip1, jm1);
    const float v_TL = u_at_face_y_d(qy, h, nx, ny, i,   j);
    const float v_TR = u_at_face_y_d(qy, h, nx, ny, ip1, j);

    const float qfy_BL = (v_BL >= 0.f) ? v_BL * h[cym]    : v_BL * h[c];
    const float qfy_BR = (v_BR >= 0.f) ? v_BR * h[cxp_ym] : v_BR * h[cxp];
    const float qfy_TL = (v_TL >= 0.f) ? v_TL * h[c]      : v_TL * h[cyp];
    const float qfy_TR = (v_TR >= 0.f) ? v_TR * h[cxp]    : v_TR * h[cxp_yp];

    const float qBarY_B  = 0.5f * (qfy_BL + qfy_BR);
    const float qBarY_T  = 0.5f * (qfy_TL + qfy_TR);
    const float uStarY_B = (qBarY_B >= 0.f) ? u_yB   : u_self;
    const float uStarY_T = (qBarY_T >  0.f) ? u_self : u_yT;

    const float h_face_sum = h[c] + h[cxp];
    const float xFluxDiff  = (qBarX_R * uStarX_R - qBarX_L * uStarX_L) / dx;
    const float yFluxDiff  = (qBarY_T * uStarY_T - qBarY_B * uStarY_B) / dx;
    const float xQDiff     = (qBarX_R - qBarX_L) / dx;
    const float yQDiff     = (qBarY_T - qBarY_B) / dx;

    const float uu = (2.f / h_face_sum) * ((xFluxDiff + yFluxDiff) - u_self * (xQDiff + yQDiff));

    float un = u_self - dt * uu - kG * dt * ((b[cxp] + h[cxp]) - (b[c] + h[c])) / dx;

    float q_out = (un >= 0.f) ? un * h[c] : un * h[cxp];

    const float hf   = 0.5f * h_face_sum;
    const float qmax = hf * faceSpeedCap_d(hf, dx, dt);
    qx_out[c]        = clampf(q_out, -qmax, qmax);
}

// Symmetric Stelling-Duinmeijer update for qy (Sim2D.cu kernel_swe_momentum_y).
__global__ void step_qy_k(const float* h, const float* b, const float* qx, const float* qy, float* qy_out, int nx,
                          int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;

    const int c = idxH(i, j, nx);
    qy_out[c]   = qy[c];

    if (j >= ny - 1) {
        qy_out[c] = 0.f;
        return;
    }

    const int cyp = idxH(i, j + 1, nx);

    if (h[c] + h[cyp] < 2.f * kDryEps) {
        qy_out[c] = 0.f;
        return;
    }

    const int im1 = clamp_x_d(i - 1, nx);
    const int ip1 = clamp_x_d(i + 1, nx);
    const int jm1 = clamp_y_d(j - 1, ny);
    const int jp1 = clamp_y_d(j + 1, ny);
    const int jp2 = clamp_y_d(j + 2, ny);

    const int cym    = idxH(i,   jm1, nx);
    const int cypp   = idxH(i,   jp2, nx);
    const int cxm    = idxH(im1, j,   nx);
    const int cxp    = idxH(ip1, j,   nx);
    const int cyp_xm = idxH(im1, jp1, nx);
    const int cyp_xp = idxH(ip1, jp1, nx);

    const float v_self = u_at_face_y_d(qy, h, nx, ny, i,   j);
    const float v_yB   = u_at_face_y_d(qy, h, nx, ny, i,   jm1);
    const float v_yT   = u_at_face_y_d(qy, h, nx, ny, i,   jp1);
    const float v_xL   = u_at_face_y_d(qy, h, nx, ny, im1, j);
    const float v_xR   = u_at_face_y_d(qy, h, nx, ny, ip1, j);

    const float qfy_B = (v_yB   >= 0.f) ? v_yB   * h[cym] : v_yB   * h[c];
    const float qfy_S = (v_self >= 0.f) ? v_self * h[c]   : v_self * h[cyp];
    const float qfy_T = (v_yT   >= 0.f) ? v_yT   * h[cyp] : v_yT   * h[cypp];

    const float qBarY_B  = 0.5f * (qfy_B + qfy_S);
    const float qBarY_T  = 0.5f * (qfy_S + qfy_T);
    const float vStarY_B = (qBarY_B >= 0.f) ? v_yB   : v_self;
    const float vStarY_T = (qBarY_T >  0.f) ? v_self : v_yT;

    const float u_LB = u_at_face_x_d(qx, h, nx, im1, j);
    const float u_RB = u_at_face_x_d(qx, h, nx, i,   j);
    const float u_LT = u_at_face_x_d(qx, h, nx, im1, jp1);
    const float u_RT = u_at_face_x_d(qx, h, nx, i,   jp1);

    const float qfx_LB = (u_LB >= 0.f) ? u_LB * h[cxm]    : u_LB * h[c];
    const float qfx_RB = (u_RB >= 0.f) ? u_RB * h[c]      : u_RB * h[cxp];
    const float qfx_LT = (u_LT >= 0.f) ? u_LT * h[cyp_xm] : u_LT * h[cyp];
    const float qfx_RT = (u_RT >= 0.f) ? u_RT * h[cyp]    : u_RT * h[cyp_xp];

    const float qBarX_L  = 0.5f * (qfx_LB + qfx_LT);
    const float qBarX_R  = 0.5f * (qfx_RB + qfx_RT);
    const float vStarX_L = (qBarX_L >= 0.f) ? v_xL   : v_self;
    const float vStarX_R = (qBarX_R >  0.f) ? v_self : v_xR;

    const float h_face_sum = h[c] + h[cyp];
    const float yFluxDiff  = (qBarY_T * vStarY_T - qBarY_B * vStarY_B) / dx;
    const float xFluxDiff  = (qBarX_R * vStarX_R - qBarX_L * vStarX_L) / dx;
    const float yQDiff     = (qBarY_T - qBarY_B) / dx;
    const float xQDiff     = (qBarX_R - qBarX_L) / dx;

    const float vv = (2.f / h_face_sum) * ((yFluxDiff + xFluxDiff) - v_self * (yQDiff + xQDiff));

    float vn = v_self - dt * vv - kG * dt * ((b[cyp] + h[cyp]) - (b[c] + h[c])) / dx;

    float q_out = (vn >= 0.f) ? vn * h[c] : vn * h[cyp];

    const float hf   = 0.5f * h_face_sum;
    const float qmax = hf * faceSpeedCap_d(hf, dx, dt);
    qy_out[c]        = clampf(q_out, -qmax, qmax);
}

// Continuity update: qx already encodes upwind h*u, so the cell divergence is
// the simple difference of right-face minus left-face fluxes (and top minus
// bottom for y). Walls contribute zero flux.
__global__ void step_h_k(const float* h, const float* qx, const float* qy, float* h_out, int nx, int ny, float dx,
                         float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;

    const int   c          = idxH(i, j, nx);
    const float qx_right   = qx[c];
    const float qx_left    = (i > 0) ? qx[idxH(i - 1, j, nx)] : 0.f;
    const float qy_top     = qy[c];
    const float qy_bottom  = (j > 0) ? qy[idxH(i, j - 1, nx)] : 0.f;

    const float divQ = (qx_right - qx_left + qy_top - qy_bottom) / dx;
    h_out[c]         = fmaxf(0.f, h[c] - dt * divQ);
}

__global__ void clamp_face_qx_k(float* qx, const float* h, int nx, int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;
    const int c = idxH(i, j, nx);
    if (i >= nx - 1) {
        qx[c] = 0.f;
        return;
    }
    const int   cxp = idxH(i + 1, j, nx);
    const float hf  = 0.5f * (h[c] + h[cxp]);
    if (hf < kDryEps) {
        qx[c] = 0.f;
        return;
    }
    const float cap  = faceSpeedCap_d(hf, dx, dt);
    const float qmax = hf * cap;
    qx[c]            = clampf(qx[c], -qmax, qmax);
}

__global__ void clamp_face_qy_k(float* qy, const float* h, int nx, int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;
    const int c = idxH(i, j, nx);
    if (j >= ny - 1) {
        qy[c] = 0.f;
        return;
    }
    const int   cyp = idxH(i, j + 1, nx);
    const float hf  = 0.5f * (h[c] + h[cyp]);
    if (hf < kDryEps) {
        qy[c] = 0.f;
        return;
    }
    const float cap  = faceSpeedCap_d(hf, dx, dt);
    const float qmax = hf * cap;
    qy[c]            = clampf(qy[c], -qmax, qmax);
}

void launchBoundary(float* d_qx, float* d_qy, const float* d_h, const float* d_terrain, int nx, int ny) {
    const int t1 = 256;
    bc_domain_qx_k<<<(ny + t1 - 1) / t1, t1>>>(d_qx, nx, ny);
    bc_domain_qy_k<<<(nx + t1 - 1) / t1, t1>>>(d_qy, nx, ny);
    SWE_POST_KERNEL();

    const dim3 tb(16, 16);
    const dim3 gb1((nx + tb.x - 1) / tb.x, (ny + tb.y - 1) / tb.y);
    bc_terrain_qx_k<<<gb1, tb>>>(d_qx, d_h, d_terrain, nx, ny);
    bc_terrain_qy_k<<<gb1, tb>>>(d_qy, d_h, d_terrain, nx, ny);
    SWE_POST_KERNEL();
}

void launchClampFaceQ(float* d_qx, float* d_qy, const float* d_h, int nx, int ny, float dx, float dt) {
    const dim3 tb(16, 16);
    const dim3 gbCell((nx + tb.x - 1) / tb.x, (ny + tb.y - 1) / tb.y);
    clamp_face_qx_k<<<gbCell, tb>>>(d_qx, d_h, nx, ny, dx, dt);
    clamp_face_qy_k<<<gbCell, tb>>>(d_qy, d_h, nx, ny, dx, dt);
    SWE_POST_KERNEL();
}

struct SweGpuBuffers {
    int nx = -1;
    int ny = -1;

    float* d_h       = nullptr;
    float* d_qx      = nullptr;
    float* d_qy      = nullptr;
    float* d_terrain = nullptr;

    float* d_qx_new = nullptr;
    float* d_qy_new = nullptr;
    float* d_h_new  = nullptr;

    ~SweGpuBuffers() {
        cudaFree(d_h);
        cudaFree(d_qx);
        cudaFree(d_qy);
        cudaFree(d_terrain);
        cudaFree(d_qx_new);
        cudaFree(d_qy_new);
        cudaFree(d_h_new);
    }

    void ensure(int nxIn, int nyIn) {
        if (nx == nxIn && ny == nyIn && d_h)
            return;

        bp_gpu::terrainCacheInvalidate();
        cudaFree(d_h);
        cudaFree(d_qx);
        cudaFree(d_qy);
        cudaFree(d_terrain);
        cudaFree(d_qx_new);
        cudaFree(d_qy_new);
        cudaFree(d_h_new);
        d_h = d_qx = d_qy = d_terrain = d_qx_new = d_qy_new = d_h_new = nullptr;

        nx = nxIn;
        ny = nyIn;
        const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);

        SWE_CUDA_CHECK(cudaSetDevice(0));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_terrain), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_new), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_new), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_new), ncell * sizeof(float)));
    }
};

static SweGpuBuffers g_swe;

void uploadGridToDevice(const Grid& g) {
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_h, g.h.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_qx, g.qx.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_qy, g.qy.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    if (!bp_gpu::sweTerrainDeviceMatchesHostCache(g.terrain.data(), ncell)) {
        SWE_CUDA_CHECK(
            cudaMemcpy(g_swe.d_terrain, g.terrain.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
        bp_gpu::noteSweTerrainH2d(g.terrain.data(), ncell);
    }
}

void downloadGridToHost(Grid& g) {
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    SWE_CUDA_CHECK(cudaMemcpy(g.h.data(), g_swe.d_h, ncell * sizeof(float), cudaMemcpyDeviceToHost));
    SWE_CUDA_CHECK(cudaMemcpy(g.qx.data(), g_swe.d_qx, ncell * sizeof(float), cudaMemcpyDeviceToHost));
    SWE_CUDA_CHECK(cudaMemcpy(g.qy.data(), g_swe.d_qy, ncell * sizeof(float), cudaMemcpyDeviceToHost));
}

void uploadTerrainOnly(const Grid& g) {
    g_swe.ensure(g.NX, g.NY);
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    if (!bp_gpu::sweTerrainDeviceMatchesHostCache(g.terrain.data(), ncell)) {
        SWE_CUDA_CHECK(
            cudaMemcpy(g_swe.d_terrain, g.terrain.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
        bp_gpu::noteSweTerrainH2d(g.terrain.data(), ncell);
    }
}

void runSweStepKernelsNoSync(int nx, int ny, float dx, float dt) {
    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, nx, ny);

    const dim3 tb(16, 16);
    const dim3 gbCell((nx + tb.x - 1) / tb.x, (ny + tb.y - 1) / tb.y);

    step_qx_k<<<gbCell, tb>>>(g_swe.d_h, g_swe.d_terrain, g_swe.d_qx, g_swe.d_qy, g_swe.d_qx_new, nx, ny, dx, dt);
    step_qy_k<<<gbCell, tb>>>(g_swe.d_h, g_swe.d_terrain, g_swe.d_qx, g_swe.d_qy, g_swe.d_qy_new, nx, ny, dx, dt);
    SWE_POST_KERNEL();

    std::swap(g_swe.d_qx, g_swe.d_qx_new);
    std::swap(g_swe.d_qy, g_swe.d_qy_new);
    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, nx, ny);

    step_h_k<<<gbCell, tb>>>(g_swe.d_h, g_swe.d_qx, g_swe.d_qy, g_swe.d_h_new, nx, ny, dx, dt);
    SWE_POST_KERNEL();
    std::swap(g_swe.d_h, g_swe.d_h_new);

    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, nx, ny);
    launchClampFaceQ(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, nx, ny, dx, dt);
}

void runBoundaryAndClampNoSync(int nx, int ny, float dx, float dt) {
    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, nx, ny);
    launchClampFaceQ(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, nx, ny, dx, dt);
}

} // namespace bp_swe_detail

void swePrefetchDeviceTerrain(const Grid& g) {
    BP_SWE_CUDA_OK(cudaSetDevice(0));
    bp_swe_detail::uploadTerrainOnly(g);
}

void sweStepGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny, float dx,
                             float dt) {
    BP_SWE_CUDA_OK(cudaSetDevice(0));
    bp_swe_detail::g_swe.ensure(nx, ny);
    bp_swe_detail::uploadTerrainOnly(gTerrainRef);
    const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_h, d_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_qx, d_qx, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_qy, d_qy, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    bp_swe_detail::runSweStepKernelsNoSync(nx, ny, dx, dt);
    BP_SWE_CUDA_OK(cudaDeviceSynchronize());
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_h, bp_swe_detail::g_swe.d_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_qx, bp_swe_detail::g_swe.d_qx, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_qy, bp_swe_detail::g_swe.d_qy, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
}

void sweApplyBoundaryGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny,
                                      float dx, float dt) {
    BP_SWE_CUDA_OK(cudaSetDevice(0));
    bp_swe_detail::g_swe.ensure(nx, ny);
    bp_swe_detail::uploadTerrainOnly(gTerrainRef);
    const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_h, d_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_qx, d_qx, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(bp_swe_detail::g_swe.d_qy, d_qy, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    bp_swe_detail::runBoundaryAndClampNoSync(nx, ny, dx, dt);
    BP_SWE_CUDA_OK(cudaDeviceSynchronize());
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_h, bp_swe_detail::g_swe.d_h, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_qx, bp_swe_detail::g_swe.d_qx, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
    BP_SWE_CUDA_OK(
        cudaMemcpy(d_qy, bp_swe_detail::g_swe.d_qy, ncell * sizeof(float), cudaMemcpyDeviceToDevice));
}

void sweDownloadGridFromDevice(Grid& g, const float* d_h, const float* d_qx, const float* d_qy) {
    BP_SWE_CUDA_OK(cudaSetDevice(0));
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    BP_SWE_CUDA_OK(cudaMemcpy(g.h.data(), d_h, ncell * sizeof(float), cudaMemcpyDeviceToHost));
    BP_SWE_CUDA_OK(cudaMemcpy(g.qx.data(), d_qx, ncell * sizeof(float), cudaMemcpyDeviceToHost));
    BP_SWE_CUDA_OK(cudaMemcpy(g.qy.data(), d_qy, ncell * sizeof(float), cudaMemcpyDeviceToHost));
}

void sweApplyBoundaryConditionsGpu(Grid& g) {
    bp_swe_detail::g_swe.ensure(g.NX, g.NY);
    bp_swe_detail::uploadGridToDevice(g);
    bp_swe_detail::runBoundaryAndClampNoSync(g.NX, g.NY, g.dx, g.dt);
    BP_SWE_CUDA_OK(cudaDeviceSynchronize());
    bp_swe_detail::downloadGridToHost(g);
}

void sweStepGpu(Grid& g) {
    bp_swe_detail::g_swe.ensure(g.NX, g.NY);
    bp_swe_detail::uploadGridToDevice(g);

    bp_swe_detail::runSweStepKernelsNoSync(g.NX, g.NY, g.dx, g.dt);
    BP_SWE_CUDA_OK(cudaDeviceSynchronize());
    bp_swe_detail::downloadGridToHost(g);
}
