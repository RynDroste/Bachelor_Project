#include "shallow_water_solver.h"
#include "wavedecomposer.h"

#include <cmath>
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
constexpr float kInvSqrt2 = 0.70710678118f;

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

    const int niters[] = {0, 8, 16, 32, 128};

    for (int niter : niters) {
        for (int step = 0; step <= 32; ++step) {
            const float lambda = 2.f + 0.5f * static_cast<float>(step);
            Grid g(kNx, kNy, kDx, kDt);
            fillCosSinPerturbation(g, lambda);

            WaveDecomposition dec;
            waveDecompose(g, kDGrad, niter, dec);

            const float wMix    = wFromDualChannel(dec.h_bar, dec.h_tilde, kNx, jrow, kDx, lambda, kH0);
            const float rel     = relativeWaveSpeedProxy(lambda, wMix);
            (void) rel;
        }
    }

    return 0;
}
