#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/gpu_terrain_h2d_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

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
// Momentum clamp: |u| <= dx/(kCflFactor*dt). Also cap so (|u|+sqrt(gh))*dt/dx <= kCflWave.
constexpr float kCflFactor = 50.f;
constexpr float kCflWave   = 0.4f;

__device__ __forceinline__ int idxH(int i, int j, int nx) { return i + j * nx; }
__device__ __forceinline__ int idxQX(int i, int j, int nx) { return i + j * (nx + 1); }
__device__ __forceinline__ int idxQY(int i, int j, int nx) { return i + j * nx; }

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float faceH_X_d(const float* h, const float* qx, int nx, int ny, int i, int j) {
    const int il = max(0, i - 1);
    const int ir = min(nx - 1, i);
    (void)ny;
    const float q = qx[idxQX(i, j, nx)];
    return (q >= 0.f) ? h[idxH(il, j, nx)] : h[idxH(ir, j, nx)];
}

__device__ __forceinline__ float faceH_Y_d(const float* h, const float* qy, int nx, int ny, int i, int j) {
    const int jd = max(0, j - 1);
    const int ju = min(ny - 1, j);
    const float q = qy[idxQY(i, j, nx)];
    return (q >= 0.f) ? h[idxH(i, jd, nx)] : h[idxH(i, ju, nx)];
}

__device__ __forceinline__ float uX_d(const float* h, const float* qx, int nx, int ny, int i, int j) {
    const float hf = faceH_X_d(h, qx, nx, ny, i, j);
    return (hf < kDryEps) ? 0.f : qx[idxQX(i, j, nx)] / hf;
}

__device__ __forceinline__ float uY_d(const float* h, const float* qy, int nx, int ny, int i, int j) {
    const float hf = faceH_Y_d(h, qy, nx, ny, i, j);
    return (hf < kDryEps) ? 0.f : qy[idxQY(i, j, nx)] / hf;
}

__device__ __forceinline__ float avgQX_d(const float* qx, int nx, int i, int j) {
    return 0.5f * (qx[idxQX(i, j, nx)] + qx[idxQX(i + 1, j, nx)]);
}

__device__ __forceinline__ float avgQY_d(const float* qy, int nx, int i, int j) {
    return 0.5f * (qy[idxQY(i, j, nx)] + qy[idxQY(i, j + 1, nx)]);
}

__device__ __forceinline__ float faceSpeedCap_d(float hf, float dx, float dt) {
    const float u_adv  = dx / (kCflFactor * dt);
    const float c      = sqrtf(fmaxf(0.f, kG * hf));
    const float u_wave = kCflWave * dx / dt - c;
    return fminf(u_adv, fmaxf(u_wave, 0.f));
}

__global__ void bc_domain_qx_k(float* qx, int nx, int ny) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny)
        return;
    qx[idxQX(0, j, nx)]  = 0.f;
    qx[idxQX(nx, j, nx)] = 0.f;
}

__global__ void bc_domain_qy_k(float* qy, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx)
        return;
    qy[idxQY(i, 0, nx)]  = 0.f;
    qy[idxQY(i, ny, nx)] = 0.f;
}

__global__ void bc_terrain_qx_k(float* qx, const float* h, const float* terrain, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= nx || j >= ny)
        return;
    const float bmax = fmaxf(terrain[idxH(i - 1, j, nx)], terrain[idxH(i, j, nx)]);
    const float wL   = terrain[idxH(i - 1, j, nx)] + h[idxH(i - 1, j, nx)];
    const float wR   = terrain[idxH(i, j, nx)] + h[idxH(i, j, nx)];
    if (bmax >= fminf(wL, wR) - kDryEps)
        qx[idxQX(i, j, nx)] = 0.f;
}

__global__ void bc_terrain_qy_k(float* qy, const float* h, const float* terrain, int nx, int ny) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j <= 0 || j >= ny)
        return;
    const float bmax = fmaxf(terrain[idxH(i, j - 1, nx)], terrain[idxH(i, j, nx)]);
    const float wD   = terrain[idxH(i, j - 1, nx)] + h[idxH(i, j - 1, nx)];
    const float wU   = terrain[idxH(i, j, nx)] + h[idxH(i, j, nx)];
    if (bmax >= fminf(wD, wU) - kDryEps)
        qy[idxQY(i, j, nx)] = 0.f;
}

__global__ void step_qx_k(const float* h, const float* qx, const float* qy, float* qx_out, int nx, int ny, float dx,
                          float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx + 1 || j >= ny)
        return;

    qx_out[idxQX(i, j, nx)] = qx[idxQX(i, j, nx)];
    if (i <= 0 || i >= nx)
        return;

    const float hf = faceH_X_d(h, qx, nx, ny, i, j);
    if (hf < kDryEps) {
        qx_out[idxQX(i, j, nx)] = 0.f;
        return;
    }

    const float umax = faceSpeedCap_d(hf, dx, dt);
    const float u    = uX_d(h, qx, nx, ny, i, j);

    const float qx_avg = avgQX_d(qx, nx, i - 1, j);
    float du_dx;
    if (qx_avg >= 0.f)
        du_dx = (u - uX_d(h, qx, nx, ny, i - 1, j)) / dx;
    else
        du_dx = (uX_d(h, qx, nx, ny, i + 1, j) - u) / dx;

    const float qy_mid = 0.5f * (qy[idxQY(max(0, i - 1), j, nx)] + qy[idxQY(min(nx - 1, i), j, nx)]);
    float du_dy;
    if (qy_mid >= 0.f) {
        const float u_d = (j > 0) ? uX_d(h, qx, nx, ny, i, j - 1) : u;
        du_dy = (u - u_d) / dx;
    } else {
        const float u_u = (j < ny - 1) ? uX_d(h, qx, nx, ny, i, j + 1) : u;
        du_dy = (u_u - u) / dx;
    }

    const float hL = h[idxH(max(0, i - 1), j, nx)];
    const float hR = h[idxH(min(nx - 1, i), j, nx)];
    float pres = 0.f;
    if (hL >= kDryEps && hR >= kDryEps)
        pres = kG * (hR - hL) / dx;

    const float advection  = (qx_avg / hf) * du_dx + (qy_mid / hf) * du_dy;
    const float qx_current = qx[idxQX(i, j, nx)];
    float qx_next          = qx_current - dt * (hf * advection + hf * pres);
    const float qmax       = hf * umax;
    qx_next                = clampf(qx_next, -qmax, qmax);
    qx_out[idxQX(i, j, nx)] = qx_next;
}

__global__ void step_qy_k(const float* h, const float* qx, const float* qy, float* qy_out, int nx, int ny, float dx,
                          float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny + 1)
        return;

    qy_out[idxQY(i, j, nx)] = qy[idxQY(i, j, nx)];
    if (j <= 0 || j >= ny)
        return;

    const float hf = faceH_Y_d(h, qy, nx, ny, i, j);
    if (hf < kDryEps) {
        qy_out[idxQY(i, j, nx)] = 0.f;
        return;
    }

    const float vmax = faceSpeedCap_d(hf, dx, dt);
    const float v    = uY_d(h, qy, nx, ny, i, j);

    const float qx_mid = 0.5f * (qx[idxQX(i, max(0, j - 1), nx)] + qx[idxQX(i, min(ny - 1, j), nx)]);
    float dv_dx;
    if (qx_mid >= 0.f) {
        const float v_l = (i > 0) ? uY_d(h, qy, nx, ny, i - 1, j) : v;
        dv_dx = (v - v_l) / dx;
    } else {
        const float v_r = (i < nx - 1) ? uY_d(h, qy, nx, ny, i + 1, j) : v;
        dv_dx = (v_r - v) / dx;
    }

    const float qy_avg = avgQY_d(qy, nx, i, j - 1);
    float dv_dy;
    if (qy_avg >= 0.f) {
        const float v_d = (j > 1) ? uY_d(h, qy, nx, ny, i, j - 1) : v;
        dv_dy = (v - v_d) / dx;
    } else {
        const float v_u = (j < ny) ? uY_d(h, qy, nx, ny, i, j + 1) : v;
        dv_dy = (v_u - v) / dx;
    }

    const float hD = h[idxH(i, max(0, j - 1), nx)];
    const float hU = h[idxH(i, min(ny - 1, j), nx)];
    float pres = 0.f;
    if (hD >= kDryEps && hU >= kDryEps)
        pres = kG * (hU - hD) / dx;

    const float advection  = (qx_mid / hf) * dv_dx + (qy_avg / hf) * dv_dy;
    const float qy_current = qy[idxQY(i, j, nx)];
    float qy_next          = qy_current - dt * (hf * advection + hf * pres);
    const float qmax       = hf * vmax;
    qy_next                = clampf(qy_next, -qmax, qmax);
    qy_out[idxQY(i, j, nx)] = qy_next;
}

__global__ void step_h_k(const float* h, const float* qx, const float* qy, float* h_out, int nx, int ny, float dx,
                         float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny)
        return;

    const float hR = faceH_X_d(h, qx, nx, ny, i + 1, j);
    const float uR = uX_d(h, qx, nx, ny, i + 1, j);
    const float hL = faceH_X_d(h, qx, nx, ny, i, j);
    const float uL = uX_d(h, qx, nx, ny, i, j);
    const float hT = faceH_Y_d(h, qy, nx, ny, i, j + 1);
    const float vT = uY_d(h, qy, nx, ny, i, j + 1);
    const float hB = faceH_Y_d(h, qy, nx, ny, i, j);
    const float vB = uY_d(h, qy, nx, ny, i, j);

    const float divQ = (hR * uR - hL * uL + hT * vT - hB * vB) / dx;
    h_out[idxH(i, j, nx)] = fmaxf(0.f, h[idxH(i, j, nx)] - dt * divQ);
}

// After h is updated, q may still imply |q|/h_face larger than the CFL cap (leapfrog mismatch); resync.
__global__ void clamp_face_qx_k(float* qx, const float* h, int nx, int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx + 1 || j >= ny)
        return;
    if (i <= 0 || i >= nx)
        return;

    const float hf = faceH_X_d(h, qx, nx, ny, i, j);
    if (hf < kDryEps) {
        qx[idxQX(i, j, nx)] = 0.f;
        return;
    }
    const float cap  = faceSpeedCap_d(hf, dx, dt);
    const float qmax = hf * cap;
    qx[idxQX(i, j, nx)] = clampf(qx[idxQX(i, j, nx)], -qmax, qmax);
}

__global__ void clamp_face_qy_k(float* qy, const float* h, int nx, int ny, float dx, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny + 1)
        return;
    if (j <= 0 || j >= ny)
        return;

    const float hf = faceH_Y_d(h, qy, nx, ny, i, j);
    if (hf < kDryEps) {
        qy[idxQY(i, j, nx)] = 0.f;
        return;
    }
    const float cap  = faceSpeedCap_d(hf, dx, dt);
    const float qmax = hf * cap;
    qy[idxQY(i, j, nx)] = clampf(qy[idxQY(i, j, nx)], -qmax, qmax);
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
    const dim3 gbQx((nx + 1 + tb.x - 1) / tb.x, (ny + tb.y - 1) / tb.y);
    const dim3 gbQy((nx + tb.x - 1) / tb.x, (ny + 1 + tb.y - 1) / tb.y);
    clamp_face_qx_k<<<gbQx, tb>>>(d_qx, d_h, nx, ny, dx, dt);
    clamp_face_qy_k<<<gbQy, tb>>>(d_qy, d_h, nx, ny, dx, dt);
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
        const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
        const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);

        SWE_CUDA_CHECK(cudaSetDevice(0));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx), nqx * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy), nqy * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_terrain), ncell * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qx_new), nqx * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_qy_new), nqy * sizeof(float)));
        SWE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_h_new), ncell * sizeof(float)));
    }
};

static SweGpuBuffers g_swe;

void uploadGridToDevice(const Grid& g) {
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    const size_t nqx   = static_cast<size_t>(g.NX + 1) * static_cast<size_t>(g.NY);
    const size_t nqy   = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY + 1);
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_h, g.h.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_qx, g.qx.data(), nqx * sizeof(float), cudaMemcpyHostToDevice));
    SWE_CUDA_CHECK(cudaMemcpy(g_swe.d_qy, g.qy.data(), nqy * sizeof(float), cudaMemcpyHostToDevice));
    if (!bp_gpu::sweTerrainDeviceMatchesHostCache(g.terrain.data(), ncell)) {
        SWE_CUDA_CHECK(
            cudaMemcpy(g_swe.d_terrain, g.terrain.data(), ncell * sizeof(float), cudaMemcpyHostToDevice));
        bp_gpu::noteSweTerrainH2d(g.terrain.data(), ncell);
    }
}

void downloadGridToHost(Grid& g) {
    const size_t ncell = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY);
    const size_t nqx   = static_cast<size_t>(g.NX + 1) * static_cast<size_t>(g.NY);
    const size_t nqy   = static_cast<size_t>(g.NX) * static_cast<size_t>(g.NY + 1);
    SWE_CUDA_CHECK(cudaMemcpy(g.h.data(), g_swe.d_h, ncell * sizeof(float), cudaMemcpyDeviceToHost));
    SWE_CUDA_CHECK(cudaMemcpy(g.qx.data(), g_swe.d_qx, nqx * sizeof(float), cudaMemcpyDeviceToHost));
    SWE_CUDA_CHECK(cudaMemcpy(g.qy.data(), g_swe.d_qy, nqy * sizeof(float), cudaMemcpyDeviceToHost));
}

} // namespace

void sweApplyBoundaryConditionsGpu(Grid& g) {
    g_swe.ensure(g.NX, g.NY);
    uploadGridToDevice(g);
    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, g.NX, g.NY);
    launchClampFaceQ(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g.NX, g.NY, g.dx, g.dt);
    SWE_CUDA_CHECK(cudaDeviceSynchronize());
    downloadGridToHost(g);
}

void sweStepGpu(Grid& g) {
    g_swe.ensure(g.NX, g.NY);
    uploadGridToDevice(g);

    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, g.NX, g.NY);

    const dim3 tb(16, 16);
    const dim3 gbQx((g.NX + 1 + tb.x - 1) / tb.x, (g.NY + tb.y - 1) / tb.y);
    const dim3 gbQy((g.NX + tb.x - 1) / tb.x, (g.NY + 1 + tb.y - 1) / tb.y);
    const dim3 gbH((g.NX + tb.x - 1) / tb.x, (g.NY + tb.y - 1) / tb.y);

    step_qx_k<<<gbQx, tb>>>(g_swe.d_h, g_swe.d_qx, g_swe.d_qy, g_swe.d_qx_new, g.NX, g.NY, g.dx, g.dt);
    step_qy_k<<<gbQy, tb>>>(g_swe.d_h, g_swe.d_qx, g_swe.d_qy, g_swe.d_qy_new, g.NX, g.NY, g.dx, g.dt);
    SWE_POST_KERNEL();

    std::swap(g_swe.d_qx, g_swe.d_qx_new);
    std::swap(g_swe.d_qy, g_swe.d_qy_new);
    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, g.NX, g.NY);

    step_h_k<<<gbH, tb>>>(g_swe.d_h, g_swe.d_qx, g_swe.d_qy, g_swe.d_h_new, g.NX, g.NY, g.dx, g.dt);
    SWE_POST_KERNEL();
    std::swap(g_swe.d_h, g_swe.d_h_new);

    launchBoundary(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g_swe.d_terrain, g.NX, g.NY);
    launchClampFaceQ(g_swe.d_qx, g_swe.d_qy, g_swe.d_h, g.NX, g.NY, g.dx, g.dt);
    SWE_CUDA_CHECK(cudaDeviceSynchronize());
    downloadGridToHost(g);
}
