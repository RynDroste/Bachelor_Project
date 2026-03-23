// =============================================================================
// Wave decomposer — Jeschke & Wojtan-style split of h, q into low-pass (bar)
// and residual (tilde). Uses free-surface H = h + bed for filtering; α from h.
// Implementation: CUDA (wave_decompose_gpu.cu); host I/O is Grid in / WaveDecomposition out.
// =============================================================================

#pragma once

#include "shallow_water_solver.h"

#include <vector>

struct WaveDecomposition {
    std::vector<float> h_bar;
    std::vector<float> h_tilde;
    std::vector<float> qx_bar;
    std::vector<float> qx_tilde;
    std::vector<float> qy_bar;
    std::vector<float> qy_tilde;
};

// d_grad_penalty: d in α = (h²/64) * exp(-d * |∇h|²).
// n_diffusion_iters: explicit diffusion substeps (paper Fig.9 uses 8/16/32/128); 0 = no diffusion.
void waveDecompose(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out);
void waveDecompose(const Grid& g, float d_grad_penalty, WaveDecomposition& out); // default 128 iterations
