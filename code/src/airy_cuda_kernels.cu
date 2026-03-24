#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

namespace {

__constant__ float c_depths[4] = {1.f, 4.f, 16.f, 64.f};

__device__ __forceinline__ float wrapped_k1d(int idx, int n, float dx) {
    const int half = n / 2;
    const float kIdx = static_cast<float>((idx <= half) ? idx : idx - n);
    return 2.f * 3.14159265f * kIdx / (static_cast<float>(n) * dx);
}

__device__ __forceinline__ float beta_fv(float k, float dx) {
    if (fabsf(k) < 1e-14f)
        return 1.f;
    const float s = sinf(0.5f * k * dx);
    return sqrtf((2.f / (k * dx)) * s);
}

__device__ __forceinline__ float omega_airy(float k, float hBar, float g, float dx) {
    if (k < 1e-20f)
        return 0.f;
    const float kh = k * fmaxf(hBar, 1e-6f);
    const float w  = sqrtf(g * k * tanhf(kh));
    const float b  = beta_fv(k, dx);
    return w / fmaxf(b, 1e-8f);
}

__device__ __forceinline__ void spectral_dhx(float kx, float dx, cufftComplex hHat, cufftComplex& out) {
    const float p  = 0.5f * kx * dx;
    const float cp = cosf(p);
    const float sp = sinf(p);
    const float hr = hHat.x;
    const float hi = hHat.y;
    const float z1r = cp * hr + sp * hi;
    const float z1i = cp * hi - sp * hr;
    out.x           = -kx * z1i;
    out.y           = kx * z1r;
}

__device__ __forceinline__ void spectral_dhy(float ky, float dy, cufftComplex hHat, cufftComplex& out) {
    const float p  = 0.5f * ky * dy;
    const float cp = cosf(p);
    const float sp = sinf(p);
    const float hr = hHat.x;
    const float hi = hHat.y;
    const float z1r = cp * hr + sp * hi;
    const float z1i = cp * hi - sp * hr;
    out.x           = -ky * z1i;
    out.y           = ky * z1r;
}

__device__ __forceinline__ void depth_bracket(float h, int& i0, int& i1, float& w) {
    if (h <= c_depths[0]) {
        i0 = i1 = 0;
        w  = 0.f;
        return;
    }
    if (h >= c_depths[3]) {
        i0 = i1 = 3;
        w  = 0.f;
        return;
    }
    for (int k = 0; k < 3; ++k) {
        if (h <= c_depths[k + 1]) {
            i0 = k;
            i1 = k + 1;
            w  = (h - c_depths[k]) / (c_depths[k + 1] - c_depths[k]);
            return;
        }
    }
    i0 = i1 = 3;
    w  = 0.f;
}

} // namespace

__global__ void airy_real_to_complex_k(const float* __restrict__ re, cufftComplex* __restrict__ out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx].x = re[idx];
    out[idx].y = 0.f;
}

__global__ void airy_spectral_qx_k(const cufftComplex* __restrict__ hatH,
                                   const cufftComplex* __restrict__ hatQx,
                                   cufftComplex* __restrict__ out,
                                   int nx,
                                   int ny,
                                   float dx,
                                   float dy,
                                   float dt,
                                   float g,
                                   float hDepth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    const float kx = wrapped_k1d(i, nx, dx);
    const float ky = wrapped_k1d(j, ny, dy);
    const float k  = sqrtf(kx * kx + ky * ky);

    cufftComplex dhx{};
    spectral_dhx(kx, dx, hatH[idx], dhx);
    const float w = omega_airy(k, hDepth, g, dx);

    const cufftComplex q = hatQx[idx];
    if (k < 1e-14f) {
        out[idx] = q;
        return;
    }
    const float c  = cosf(w * dt);
    const float s  = sinf(w * dt);
    const float rk = w / (k * k);
    out[idx].x = c * q.x - s * rk * dhx.x;
    out[idx].y = c * q.y - s * rk * dhx.y;
}

__global__ void airy_spectral_qy_k(const cufftComplex* __restrict__ hatH,
                                   const cufftComplex* __restrict__ hatQy,
                                   cufftComplex* __restrict__ out,
                                   int nx,
                                   int ny,
                                   float dx,
                                   float dy,
                                   float dt,
                                   float g,
                                   float hDepth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    const float kx = wrapped_k1d(i, nx, dx);
    const float ky = wrapped_k1d(j, ny, dy);
    const float k  = sqrtf(kx * kx + ky * ky);

    cufftComplex dhy{};
    spectral_dhy(ky, dy, hatH[idx], dhy);
    const float w = omega_airy(k, hDepth, g, dx);

    const cufftComplex q = hatQy[idx];
    if (k < 1e-14f) {
        out[idx] = q;
        return;
    }
    const float c  = cosf(w * dt);
    const float s  = sinf(w * dt);
    const float rk = w / (k * k);
    out[idx].x = c * q.x - s * rk * dhy.x;
    out[idx].y = c * q.y - s * rk * dhy.y;
}

__global__ void airy_cpx_to_real_scaled_k(const cufftComplex* __restrict__ in,
                                          float* __restrict__ out,
                                          float scale,
                                          int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    out[idx] = in[idx].x * scale;
}

__global__ void airy_blend_k(const float* __restrict__ sx0,
                             const float* __restrict__ sx1,
                             const float* __restrict__ sx2,
                             const float* __restrict__ sx3,
                             const float* __restrict__ sy0,
                             const float* __restrict__ sy1,
                             const float* __restrict__ sy2,
                             const float* __restrict__ sy3,
                             const float* __restrict__ h_bar,
                             float* __restrict__ qxc,
                             float* __restrict__ qyc,
                             int nx,
                             int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const float hb = h_bar[idx];
    int         i0, i1;
    float       w{};
    depth_bracket(hb, i0, i1, w);

    const float* px0 = sx0 + idx;
    const float* px1 = sx1 + idx;
    const float* px2 = sx2 + idx;
    const float* px3 = sx3 + idx;
    const float* py0 = sy0 + idx;
    const float* py1 = sy1 + idx;
    const float* py2 = sy2 + idx;
    const float* py3 = sy3 + idx;

    auto sx_at = [&](int d) -> float {
        switch (d) {
        case 0:
            return *px0;
        case 1:
            return *px1;
        case 2:
            return *px2;
        default:
            return *px3;
        }
    };
    auto sy_at = [&](int d) -> float {
        switch (d) {
        case 0:
            return *py0;
        case 1:
            return *py1;
        case 2:
            return *py2;
        default:
            return *py3;
        }
    };

    float qxC = sx_at(i0);
    float qyC = sy_at(i0);
    if (i0 != i1) {
        qxC = (1.f - w) * sx_at(i0) + w * sx_at(i1);
        qyC = (1.f - w) * sy_at(i0) + w * sy_at(i1);
    }
    qxc[idx] = qxC;
    qyc[idx] = qyC;
}

void airy_cuda_launch_real_to_complex(const float* d_re, cufftComplex* d_out, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_real_to_complex_k<<<blocks, threads, 0, stream>>>(d_re, d_out, n);
}

void airy_cuda_launch_spectral_qx(const cufftComplex* hatH,
                                  const cufftComplex* hatQx,
                                  cufftComplex* out,
                                  int nx,
                                  int ny,
                                  float dx,
                                  float dy,
                                  float dt,
                                  float g,
                                  float hDepth,
                                  cudaStream_t stream) {
    const int n       = nx * ny;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_spectral_qx_k<<<blocks, threads, 0, stream>>>(hatH, hatQx, out, nx, ny, dx, dy, dt, g, hDepth);
}

void airy_cuda_launch_spectral_qy(const cufftComplex* hatH,
                                  const cufftComplex* hatQy,
                                  cufftComplex* out,
                                  int nx,
                                  int ny,
                                  float dx,
                                  float dy,
                                  float dt,
                                  float g,
                                  float hDepth,
                                  cudaStream_t stream) {
    const int n       = nx * ny;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_spectral_qy_k<<<blocks, threads, 0, stream>>>(hatH, hatQy, out, nx, ny, dx, dy, dt, g, hDepth);
}

void airy_cuda_launch_cpx_to_real_scaled(const cufftComplex* d_in, float* d_out, float scale, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_cpx_to_real_scaled_k<<<blocks, threads, 0, stream>>>(d_in, d_out, scale, n);
}

void airy_cuda_launch_blend(const float* d_sx0,
                            const float* d_sx1,
                            const float* d_sx2,
                            const float* d_sx3,
                            const float* d_sy0,
                            const float* d_sy1,
                            const float* d_sy2,
                            const float* d_sy3,
                            const float* d_h_bar,
                            float* d_qxc,
                            float* d_qyc,
                            int nx,
                            int ny,
                            cudaStream_t stream) {
    const int n       = nx * ny;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_blend_k<<<blocks, threads, 0, stream>>>(d_sx0, d_sx1, d_sx2, d_sx3, d_sy0, d_sy1, d_sy2, d_sy3, d_h_bar, d_qxc,
                                                 d_qyc, nx, ny);
}

__global__ void airy_face_to_cell_k(const float* __restrict__ qx_face,
                                    const float* __restrict__ qy_face,
                                    float* __restrict__ qxc,
                                    float* __restrict__ qyc,
                                    int nx,
                                    int ny) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n   = nx * ny;
    if (idx >= n)
        return;
    const int i = idx % nx;
    const int j = idx / nx;
    const int iq = i + j * (nx + 1);
    qxc[idx]     = 0.5f * (qx_face[iq] + qx_face[iq + 1]);
    const int ir = idx;
    qyc[idx]     = 0.5f * (qy_face[ir] + qy_face[ir + nx]);
}

__global__ void airy_cell_to_qx_faces_k(const float* __restrict__ qxc, float* __restrict__ qx_face, int nx, int ny) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int nfaces = (nx + 1) * ny;
    if (tid >= nfaces)
        return;
    const int fi = tid % (nx + 1);
    const int fj = tid / (nx + 1);
    if (fi == 0)
        qx_face[tid] = qxc[0 + fj * nx];
    else if (fi == nx)
        qx_face[tid] = qxc[(nx - 1) + fj * nx];
    else
        qx_face[tid] = 0.5f * (qxc[(fi - 1) + fj * nx] + qxc[fi + fj * nx]);
}

__global__ void airy_cell_to_qy_faces_k(const float* __restrict__ qyc, float* __restrict__ qy_face, int nx, int ny) {
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int nfaces = nx * (ny + 1);
    if (tid >= nfaces)
        return;
    const int i      = tid % nx;
    const int j_face = tid / nx;
    if (j_face == 0)
        qy_face[tid] = qyc[0 * nx + i];
    else if (j_face == ny)
        qy_face[tid] = qyc[(ny - 1) * nx + i];
    else
        qy_face[tid] = 0.5f * (qyc[(j_face - 1) * nx + i] + qyc[j_face * nx + i]);
}

void airy_cuda_launch_face_to_cell(const float* d_qx_face,
                                   const float* d_qy_face,
                                   float* d_qxc,
                                   float* d_qyc,
                                   int nx,
                                   int ny,
                                   cudaStream_t stream) {
    const int n       = nx * ny;
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;
    airy_face_to_cell_k<<<blocks, threads, 0, stream>>>(d_qx_face, d_qy_face, d_qxc, d_qyc, nx, ny);
}

void airy_cuda_launch_cell_to_qx_faces(const float* d_qxc, float* d_qx_face, int nx, int ny, cudaStream_t stream) {
    const int nfaces  = (nx + 1) * ny;
    const int threads = 256;
    const int blocks  = (nfaces + threads - 1) / threads;
    airy_cell_to_qx_faces_k<<<blocks, threads, 0, stream>>>(d_qxc, d_qx_face, nx, ny);
}

void airy_cuda_launch_cell_to_qy_faces(const float* d_qyc, float* d_qy_face, int nx, int ny, cudaStream_t stream) {
    const int nfaces  = nx * (ny + 1);
    const int threads = 256;
    const int blocks  = (nfaces + threads - 1) / threads;
    airy_cell_to_qy_faces_k<<<blocks, threads, 0, stream>>>(d_qyc, d_qy_face, nx, ny);
}
