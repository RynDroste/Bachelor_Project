#pragma once

#include "solver_pipeline/shallow_water_solver.h"

#include <vector>

struct WaveDecomposition {
    std::vector<float> h_bar;
    std::vector<float> h_tilde;
    std::vector<float> qx_bar;
    std::vector<float> qx_tilde;
    std::vector<float> qy_bar;
    std::vector<float> qy_tilde;
};

// d_grad_penalty: d in alpha = (h^2/64) * exp(-d * |grad h|^2).
void waveDecompose(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out);
void waveDecompose(const Grid& g, float d_grad_penalty, WaveDecomposition& out); // default 128 iterations
