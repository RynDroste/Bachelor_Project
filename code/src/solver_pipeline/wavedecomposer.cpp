#include "solver_pipeline/wavedecomposer.h"

#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/wave_decompose_gpu.hpp"

void waveDecompose(const Grid& g, float d_grad_penalty, WaveDecomposition& out) {
    waveDecompose(g, d_grad_penalty, 128, out);
}

void waveDecompose(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out) {
    waveDecomposeGpu(g, d_grad_penalty, n_diffusion_iters, out);
}
