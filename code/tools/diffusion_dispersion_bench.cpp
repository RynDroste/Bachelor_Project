// 生成 J&W Fig.9 风格数据：不同扩散迭代次数下，按波长扫描的相对相速代理量。
// 输出 CSV: niter,wavelength_m,theoretical_phase_speed_m_s,relative_wave_speed
// theoretical_phase_speed_m_s: Airy 线性理论相速 c=ω/k（水深 h0），作相对波速归一化基准
// 初值 h = h0 + (ε/√2)(cos kx + sin kx)，k=2π/λ：λ=2Δx 时 cos 在体心为 0 但 sin 非 0，场非退化。
// 双通道投影：A_c=(2/Nx)Σ(f−f_ref)cos(kx)，A_s=(2/Nx)Σ(f−f_ref)sin(kx)，R=√(A_c²+A_s²)；w=R̄/(R̄+R̃)。

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
constexpr float kInvSqrt2 = 0.70710678118f; // 使 (cos+sin)/√2 与幅值 ε 的单 cos 同量级 RMS

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

} // namespace

int main() {
    const int jrow = kNy / 2;
    std::printf("niter,wavelength_m,theoretical_phase_speed_m_s,relative_wave_speed\n");

    const int niters[] = {8, 16, 32, 128};
    for (int niter : niters) {
        for (int step = 0; step <= 32; ++step) {
            const float lambda = 2.f + 0.5f * static_cast<float>(step);
            Grid g(kNx, kNy, kDx, kDt);
            for (int j = 0; j < kNy; ++j) {
                for (int i = 0; i < kNx; ++i) {
                    g.B(i, j) = 0.f;
                    const float x  = (static_cast<float>(i) + 0.5f) * kDx;
                    const float kx = kTwoPi * x / lambda;
                    g.H(i, j)      = kH0 + kAmp * kInvSqrt2 * (std::cos(kx) + std::sin(kx));
                }
            }

            WaveDecomposition dec;
            waveDecompose(g, kDGrad, niter, dec);

            const float wMix = wFromDualChannel(dec.h_bar, dec.h_tilde, kNx, jrow, kDx, lambda, kH0);
            const float rel  = relativeWaveSpeedProxy(lambda, wMix);
            const float cTheory = airyPhaseSpeed(lambda, kH0);
            std::printf("%d,%.3f,%.6f,%.6f\n", niter, static_cast<double>(lambda),
                        static_cast<double>(cTheory), static_cast<double>(rel));
        }
    }
    return 0;
}
