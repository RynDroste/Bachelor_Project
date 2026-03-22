#include "airy_fftw.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <fftw3.h>

namespace {

constexpr float kPi        = 3.14159265358979323846f;
constexpr float kDepths[4] = {1.f, 4.f, 16.f, 64.f};

inline float wrappedK1D(int idx, int n, float dx) {
    const int half = n / 2;
    const float kIdx = static_cast<float>((idx <= half) ? idx : idx - n);
    return 2.f * kPi * kIdx / (static_cast<float>(n) * dx);
}

// Appendix B, Eq. (27) — FV staggered divergence correction; ω ← ω/β
inline float betaFV(float k, float dx) {
    if (std::fabs(k) < 1e-14f)
        return 1.f;
    const float s = std::sin(0.5f * k * dx);
    return std::sqrt((2.f / (k * dx)) * s);
}

inline float omegaAiry(float k, float hBar, float g, float dx) {
    if (k < 1e-20f)
        return 0.f;
    const float kh = k * std::max(hBar, 1e-6f);
    const float w  = std::sqrt(g * k * std::tanh(kh));
    const float b  = betaFV(k, dx);
    return w / std::max(b, 1e-8f);
}

// For h_bar in [d0,d1], w = 0 at d0, 1 at d1
inline void depthBracket(float h, int& i0, int& i1, float& w) {
    if (h <= kDepths[0]) {
        i0 = i1 = 0;
        w  = 0.f;
        return;
    }
    if (h >= kDepths[3]) {
        i0 = i1 = 3;
        w  = 0.f;
        return;
    }
    for (int k = 0; k < 3; ++k) {
        if (h <= kDepths[k + 1]) {
            i0 = k;
            i1 = k + 1;
            w  = (h - kDepths[k]) / (kDepths[k + 1] - kDepths[k]);
            return;
        }
    }
    i0 = i1 = 3;
    w  = 0.f;
}

// dhx_hat = i * kx * exp(-i kx dx/2) * h_hat  (qx / x-faces)
inline void spectralDHx(float kx, float dx, fftwf_complex hHat, fftwf_complex out) {
    const float p  = 0.5f * kx * dx;
    const float cp = std::cos(p);
    const float sp = std::sin(p);
    const float hr = hHat[0];
    const float hi = hHat[1];
    const float z1r = cp * hr + sp * hi;
    const float z1i = cp * hi - sp * hr;
    out[0]          = -kx * z1i;
    out[1]          = kx * z1r;
}

// dhy_hat = i * ky * exp(-i ky dy/2) * h_hat
inline void spectralDHy(float ky, float dy, fftwf_complex hHat, fftwf_complex out) {
    const float p  = 0.5f * ky * dy;
    const float cp = std::cos(p);
    const float sp = std::sin(p);
    const float hr = hHat[0];
    const float hi = hHat[1];
    const float z1r = cp * hr + sp * hi;
    const float z1i = cp * hi - sp * hr;
    out[0]          = -ky * z1i;
    out[1]          = ky * z1r;
}

} // namespace

struct AiryEWaveFFTW::Impl {
    int   nx = 0;
    int   ny = 0;
    float dx = 1.f;

    fftwf_complex* hatH = nullptr;
    fftwf_complex* hatQx = nullptr;
    fftwf_complex* hatQy = nullptr;
    fftwf_complex* specWork = nullptr;

    fftwf_plan planFwd = nullptr;
    fftwf_plan planInv = nullptr;

    std::vector<float> qxc;
    std::vector<float> qyc;
    std::vector<float> spatialQx[4];
    std::vector<float> spatialQy[4];

    ~Impl() {
        if (planFwd)
            fftwf_destroy_plan(planFwd);
        if (planInv)
            fftwf_destroy_plan(planInv);
        fftwf_free(hatH);
        fftwf_free(hatQx);
        fftwf_free(hatQy);
        fftwf_free(specWork);
    }
};

AiryEWaveFFTW::AiryEWaveFFTW(int nx, int ny, float dx)
    : nx_(nx)
    , ny_(ny)
    , dx_(dx)
    , impl_(std::make_unique<Impl>()) {
    impl_->nx = nx;
    impl_->ny = ny;
    impl_->dx = dx;
    const size_t n = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    impl_->hatH     = static_cast<fftwf_complex*>(fftwf_alloc_complex(n));
    impl_->hatQx    = static_cast<fftwf_complex*>(fftwf_alloc_complex(n));
    impl_->hatQy    = static_cast<fftwf_complex*>(fftwf_alloc_complex(n));
    impl_->specWork = static_cast<fftwf_complex*>(fftwf_alloc_complex(n));

    impl_->qxc.resize(n);
    impl_->qyc.resize(n);
    for (int d = 0; d < 4; ++d) {
        impl_->spatialQx[d].resize(n);
        impl_->spatialQy[d].resize(n);
    }

    impl_->planFwd = fftwf_plan_dft_2d(ny, nx, impl_->hatH, impl_->hatH, FFTW_FORWARD, FFTW_ESTIMATE);
    impl_->planInv = fftwf_plan_dft_2d(ny, nx, impl_->specWork, impl_->specWork, FFTW_BACKWARD, FFTW_ESTIMATE);
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
    const size_t n   = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const float invN = 1.f / static_cast<float>(n);

    // Cell-centered q from staggered tilde fluxes
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int iq = i + j * (nx + 1);
            im.qxc[j * nx + i] = 0.5f * (qx_tilde[iq] + qx_tilde[iq + 1]);
            const int ir = i + j * nx;
            im.qyc[j * nx + i] = 0.5f * (qy_tilde[ir] + qy_tilde[ir + nx]);
        }
    }

    // ĥ
    for (size_t k = 0; k < n; ++k) {
        im.hatH[k][0] = h_tilde_sym[k];
        im.hatH[k][1] = 0.f;
    }
    fftwf_execute_dft(im.planFwd, im.hatH, im.hatH);

    // q̂x, q̂y
    for (size_t k = 0; k < n; ++k) {
        im.hatQx[k][0] = im.qxc[k];
        im.hatQx[k][1] = 0.f;
        im.hatQy[k][0] = im.qyc[k];
        im.hatQy[k][1] = 0.f;
    }
    fftwf_execute_dft(im.planFwd, im.hatQx, im.hatQx);
    fftwf_execute_dft(im.planFwd, im.hatQy, im.hatQy);

    fftwf_complex dhx{};
    fftwf_complex dhy{};

    for (int d = 0; d < 4; ++d) {
        const float hDepth = kDepths[d];
        for (int j = 0; j < ny; ++j) {
            const float ky = wrappedK1D(j, ny, dy);
            for (int i = 0; i < nx; ++i) {
                const float kx = wrappedK1D(i, nx, dx);
                const float k  = std::sqrt(kx * kx + ky * ky);
                const int   idx = j * nx + i;

                spectralDHx(kx, dx, im.hatH[idx], dhx);

                const float w = omegaAiry(k, hDepth, g, dx);

                if (k < 1e-14f) {
                    im.specWork[idx][0] = im.hatQx[idx][0];
                    im.specWork[idx][1] = im.hatQx[idx][1];
                } else {
                    const float c  = std::cos(w * dt);
                    const float s  = std::sin(w * dt);
                    const float rk = w / (k * k);
                    im.specWork[idx][0] = c * im.hatQx[idx][0] - s * rk * dhx[0];
                    im.specWork[idx][1] = c * im.hatQx[idx][1] - s * rk * dhx[1];
                }
            }
        }
        fftwf_execute_dft(im.planInv, im.specWork, im.specWork);
        for (size_t kk = 0; kk < n; ++kk)
            im.spatialQx[d][kk] = im.specWork[kk][0] * invN;

        for (int j = 0; j < ny; ++j) {
            const float ky = wrappedK1D(j, ny, dy);
            for (int i = 0; i < nx; ++i) {
                const float kx = wrappedK1D(i, nx, dx);
                const float k  = std::sqrt(kx * kx + ky * ky);
                const int   idx = j * nx + i;

                spectralDHy(ky, dy, im.hatH[idx], dhy);

                const float w = omegaAiry(k, hDepth, g, dx);

                if (k < 1e-14f) {
                    im.specWork[idx][0] = im.hatQy[idx][0];
                    im.specWork[idx][1] = im.hatQy[idx][1];
                } else {
                    const float c  = std::cos(w * dt);
                    const float s  = std::sin(w * dt);
                    const float rk = w / (k * k);
                    im.specWork[idx][0] = c * im.hatQy[idx][0] - s * rk * dhy[0];
                    im.specWork[idx][1] = c * im.hatQy[idx][1] - s * rk * dhy[1];
                }
            }
        }
        fftwf_execute_dft(im.planInv, im.specWork, im.specWork);
        for (size_t kk = 0; kk < n; ++kk)
            im.spatialQy[d][kk] = im.specWork[kk][0] * invN;
    }

    // Spatial blend by local h̄ (论文 §4.3)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int   idx  = j * nx + i;
            const float hb   = h_bar[idx];
            int         i0, i1;
            float       w{};
            depthBracket(hb, i0, i1, w);
            const float qxC =
                (i0 == i1) ? im.spatialQx[i0][idx]
                           : (1.f - w) * im.spatialQx[i0][idx] + w * im.spatialQx[i1][idx];
            const float qyC =
                (i0 == i1) ? im.spatialQy[i0][idx]
                           : (1.f - w) * im.spatialQy[i0][idx] + w * im.spatialQy[i1][idx];
            im.qxc[idx] = qxC;
            im.qyc[idx] = qyC;
        }
    }

    // Scatter cell q → staggered faces
    for (int j = 0; j < ny; ++j) {
        qx_tilde[0 + j * (nx + 1)] = im.qxc[j * nx + 0];
        for (int i = 1; i < nx; ++i) {
            const int iq = i + j * (nx + 1);
            qx_tilde[iq] = 0.5f * (im.qxc[j * nx + (i - 1)] + im.qxc[j * nx + i]);
        }
        qx_tilde[nx + j * (nx + 1)] = im.qxc[j * nx + (nx - 1)];
    }

    for (int i = 0; i < nx; ++i) {
        qy_tilde[i + 0 * nx] = im.qyc[0 * nx + i];
        for (int j = 1; j < ny; ++j) {
            const int ir = i + j * nx;
            qy_tilde[ir] = 0.5f * (im.qyc[(j - 1) * nx + i] + im.qyc[j * nx + i]);
        }
        qy_tilde[i + ny * nx] = im.qyc[(ny - 1) * nx + i];
    }
}
