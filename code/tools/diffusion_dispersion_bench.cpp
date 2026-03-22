// J&W Fig.9-style sweep: relative phase-speed proxy vs wavelength for several diffusion iteration counts.
// CSV: niter,wavelength_m,theoretical_phase_speed_m_s,relative_wave_speed
// theoretical_phase_speed_m_s: Airy c = omega/k at depth h0 (normalization for relative_wave_speed).
// IC: h = h0 + (eps/sqrt(2))(cos kx + sin kx); at lambda = 2*dx, cos vanishes at cell centers but sin does not.
// Dual-channel: Ac, As from cos/sin projections; R = hypot(Ac,As); w = Rbar/(Rbar+Rtilde).

#include "shallow_water_solver.h"
#include "wavedecomposer.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr float kG        = 9.81f;
constexpr float kH0       = 4.f;
constexpr float kDx       = 1.f;
constexpr float kDt       = 1.f / 60.f;
constexpr float kDGrad    = 0.01f;
constexpr float kAmp      = 1e-3f;
constexpr int   kNx       = 512;
constexpr int   kNy       = 8;
constexpr float kPi       = 3.14159265358979323846f;
constexpr float kTwoPi   = 2.f * kPi;
constexpr float kInvSqrt2 = 0.70710678118f; // same RMS scale as single cos with amplitude eps

float airyPhaseSpeed(float wavelength, float h0) {
    if (wavelength < 1e-6f)
        return std::sqrt(kG * h0);
    const float k = kTwoPi / wavelength;
    const float kh = k * std::max(h0, 1e-6f);
    return std::sqrt(kG * std::tanh(kh) / k);
}

float rowMean(const std::vector<float>& f, int nx, int jrow) {
    double s = 0.0;
    for (int i = 0; i < nx; ++i)
        s += static_cast<double>(f[i + jrow * nx]);
    return static_cast<float>(s / static_cast<double>(nx));
}

float projectModeAmpCosSin(const std::vector<float>& f, int nx, int jrow, float dx, float lambda, float ref) {
    const double k = static_cast<double>(kTwoPi) / static_cast<double>(lambda);
    double         accC = 0.0;
    double         accS = 0.0;
    for (int i = 0; i < nx; ++i) {
        const double x = (static_cast<double>(i) + 0.5) * static_cast<double>(dx);
        const double t = k * x;
        const double v = static_cast<double>(f[i + jrow * nx] - ref);
        accC += v * std::cos(t);
        accS += v * std::sin(t);
    }
    const double scale = 2.0 / static_cast<double>(nx);
    return static_cast<float>(std::hypot(scale * accC, scale * accS));
}

float wFromDualChannel(const std::vector<float>& hBar, const std::vector<float>& hTilde, int nx, int jrow, float dx,
                       float lambda, float h0) {
    const float rbar   = projectModeAmpCosSin(hBar, nx, jrow, dx, lambda, h0);
    const float mtilde = rowMean(hTilde, nx, jrow);
    const float rtilde = projectModeAmpCosSin(hTilde, nx, jrow, dx, lambda, mtilde);
    const float den    = rbar + rtilde + 1e-20f;
    return rbar / den;
}

float relativeWaveSpeedProxy(float lambda, float wBar) {
    const float cAiry = airyPhaseSpeed(lambda, kH0);
    const float cSW   = std::sqrt(kG * kH0);
    if (cAiry < 1e-8f)
        return 1.f;
    const float w = std::clamp(wBar, 0.f, 1.f);
    return w * (cSW / cAiry) + (1.f - w) * 1.f;
}

// Matches wavedecomposer.cpp computeAlphaFromH (flat h => alpha = h^2/64)
float alphaCellLikeDecomposer(const Grid& g, float d_grad_penalty, int i, int j) {
    constexpr float kEps = 1e-8f;
    const float     dx   = g.dx;
    int               im   = std::max(0, i - 1);
    int               ip   = std::min(g.NX - 1, i + 1);
    int               jm   = std::max(0, j - 1);
    int               jp   = std::min(g.NY - 1, j + 1);
    float             denom_x = float(ip - im) * dx;
    float             denom_y = float(jp - jm) * dx;
    float             grad_h_x  = (denom_x > 0.f) ? (g.H(ip, j) - g.H(im, j)) / denom_x : 0.f;
    float             grad_h_y  = (denom_y > 0.f) ? (g.H(i, jp) - g.H(i, jm)) / denom_y : 0.f;
    float             grad_h_sq = grad_h_x * grad_h_x + grad_h_y * grad_h_y;
    float             h         = g.H(i, j);
    return (h * h / 64.f) * std::exp(-d_grad_penalty * grad_h_sq);
}

float barModeEnergyFraction(const std::vector<float>& hBar, const std::vector<float>& hTilde, int nx, int jrow,
                            float dx, float lambda, float h0) {
    const float rbar   = projectModeAmpCosSin(hBar, nx, jrow, dx, lambda, h0);
    const float mtilde = rowMean(hTilde, nx, jrow);
    const float rtilde = projectModeAmpCosSin(hTilde, nx, jrow, dx, lambda, mtilde);
    const float eb     = rbar * rbar;
    const float et     = rtilde * rtilde;
    return eb / (eb + et + 1e-30f);
}

void fillCosSinPerturbation(Grid& g, float lambda) {
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            g.B(i, j) = 0.f;
            const float x  = (static_cast<float>(i) + 0.5f) * g.dx;
            const float kx = kTwoPi * x / lambda;
            g.H(i, j)      = kH0 + kAmp * kInvSqrt2 * (std::cos(kx) + std::sin(kx));
        }
    }
}

} // namespace

int main() {
    const int jrow = kNy / 2;
    std::printf("niter,wavelength_m,theoretical_phase_speed_m_s,relative_wave_speed\n");

    const int niters[] = {0, 8, 16, 32, 128};
    float     maxAbsErr128 = 0.f;
    float     lambdaAtMax128 = 0.f;

    for (int niter : niters) {
        for (int step = 0; step <= 32; ++step) {
            const float lambda = 2.f + 0.5f * static_cast<float>(step);
            Grid g(kNx, kNy, kDx, kDt);
            fillCosSinPerturbation(g, lambda);

            WaveDecomposition dec;
            waveDecompose(g, kDGrad, niter, dec);

            const float wMix    = wFromDualChannel(dec.h_bar, dec.h_tilde, kNx, jrow, kDx, lambda, kH0);
            const float rel     = relativeWaveSpeedProxy(lambda, wMix);
            const float cTheory = airyPhaseSpeed(lambda, kH0);
            std::printf("%d,%.3f,%.6f,%.6f\n", niter, static_cast<double>(lambda),
                        static_cast<double>(cTheory), static_cast<double>(rel));

            if (niter == 128) {
                const float e = std::fabs(rel - 1.f);
                if (e > maxAbsErr128) {
                    maxAbsErr128   = e;
                    lambdaAtMax128 = lambda;
                }
            }
        }
    }

    // stderr summary (reference magnitudes from paper; not an automated test threshold)
    Grid gFlat(kNx, kNy, kDx, kDt);
    for (int j = 0; j < kNy; ++j) {
        for (int i = 0; i < kNx; ++i) {
            gFlat.B(i, j) = 0.f;
            gFlat.H(i, j) = kH0;
        }
    }
    const int   ic      = kNx / 2;
    const float alpha0  = alphaCellLikeDecomposer(gFlat, kDGrad, ic, jrow);
    const float alphaTh = kH0 * kH0 / 64.f;

    const float lambdaCut = kTwoPi * kH0;
    Grid        gCut(kNx, kNy, kDx, kDt);
    fillCosSinPerturbation(gCut, lambdaCut);
    WaveDecomposition decCut;
    waveDecompose(gCut, kDGrad, 128, decCut);
    const float fracBar128 = barModeEnergyFraction(decCut.h_bar, decCut.h_tilde, kNx, jrow, kDx, lambdaCut, kH0);

    // lambda in [2,30] m: at niter=128, wavelength whose bar mode energy fraction is closest to 0.5
    float bestHalfLam = 0.f;
    float bestHalfErr = 1e10f;
    for (int step = 0; step <= 56; ++step) {
        const float lambda = 2.f + 0.5f * static_cast<float>(step);
        Grid        gg(kNx, kNy, kDx, kDt);
        fillCosSinPerturbation(gg, lambda);
        WaveDecomposition dd;
        waveDecompose(gg, kDGrad, 128, dd);
        const float f = barModeEnergyFraction(dd.h_bar, dd.h_tilde, kNx, jrow, kDx, lambda, kH0);
        const float e = std::fabs(f - 0.5f);
        if (e < bestHalfErr) {
            bestHalfErr = e;
            bestHalfLam = lambda;
        }
    }

    Grid gShort(kNx, kNy, kDx, kDt);
    fillCosSinPerturbation(gShort, 2.f);
    WaveDecomposition d0;
    waveDecompose(gShort, kDGrad, 0, d0);
    const float w0     = wFromDualChannel(d0.h_bar, d0.h_tilde, kNx, jrow, kDx, 2.f, kH0);
    const float rel0_2 = relativeWaveSpeedProxy(2.f, w0);

    std::fprintf(stderr,
                 "alpha check (flat h=h0, cell center): alpha=%.6g  h^2/64=%.6g  rel err %.2e\n",
                 static_cast<double>(alpha0), static_cast<double>(alphaTh),
                 static_cast<double>(std::fabs(alpha0 - alphaTh) / alphaTh));
    std::fprintf(stderr,
                 "at lambda=2*pi*h ~ %.4f m: bar energy fraction Rb^2/(Rb^2+Rt^2) (niter=128) -> %.4f\n"
                 "  (paper 'half-half' is continuous cutoff ~2*pi*h; discrete 128-step projection need not be 0.5)\n",
                 static_cast<double>(lambdaCut), static_cast<double>(fracBar128));
    std::fprintf(stderr,
                 "niter=128: lambda* in [2,30]m closest to half bar energy: %.3f m (|f-0.5|=%.4f)\n",
                 static_cast<double>(bestHalfLam), static_cast<double>(bestHalfErr));
    std::fprintf(stderr,
                 "niter=128: max|relative_wave_speed-1| = %.4f (~%.2f%%) at lambda=%.3f m (paper Fig.9 ~8%% scale)\n",
                 static_cast<double>(maxAbsErr128), static_cast<double>(100.f * maxAbsErr128),
                 static_cast<double>(lambdaAtMax128));
    std::fprintf(stderr, "niter=0, lambda=2 m: relative_wave_speed=%.4f (short waves should be >> 1)\n",
                 static_cast<double>(rel0_2));

    return 0;
}
