// Jeschke & Wojtan 2023 — Algorithm 2: Airy surface waves via FFT + exponential integrator (eWave).
// 2D, cell-centered spectral ops; qx/qy staggered ↔ cell average for FFT, then scatter back.

#pragma once

#include <memory>

class AiryEWaveFFTW {
public:
    AiryEWaveFFTW(int nx, int ny, float dx);
    ~AiryEWaveFFTW();

    AiryEWaveFFTW(const AiryEWaveFFTW&)            = delete;
    AiryEWaveFFTW& operator=(const AiryEWaveFFTW&) = delete;
    AiryEWaveFFTW(AiryEWaveFFTW&&)                 = delete;
    AiryEWaveFFTW& operator=(AiryEWaveFFTW&&)      = delete;

    // h_tilde_sym: 与 q 对齐的时刻 t 上的 tilde 水深（论文用 (h^{t-Δt/2}+h^{t+Δt/2})/2）
    // h_bar: 每格平滑水深，用于 ω=√(gk tanh(kh̄)) 的多档插值
    // qx_tilde / qy_tilde: 交错网格通量，原地更新
    void step(float dt, float g,
              const float* h_tilde_sym,
              const float* h_bar,
              float*       qx_tilde,
              float*       qy_tilde);

    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    int   nx_;
    int   ny_;
    float dx_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
