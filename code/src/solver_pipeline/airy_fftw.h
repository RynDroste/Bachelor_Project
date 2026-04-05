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

    // h_tilde_sym: tilde depth at time t aligned with q
    // h_bar: smoothed depth per cell for piecewise omega = sqrt(g k tanh(k h_bar))
    // qx_tilde / qy_tilde: staggered-face fluxes, updated in place
    void step(float dt, float g,
              const float* h_tilde_sym,
              const float* h_bar,
              float*       qx_tilde,
              float*       qy_tilde);

    // Same physics as step(); all pointers are device (CUDA). nx/ny/dx must match constructor grid.
    void stepDevice(float dt, float g, const float* d_h_tilde_sym, const float* d_h_bar, float* d_qx_tilde,
                    float* d_qy_tilde);

    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    int   nx_;
    int   ny_;
    float dx_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
