#pragma once

#include <cstddef>

struct Grid;
struct WaveDecomposition;

void waveDecomposeGpu(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out);

struct WaveDecompGpuPtrs {
    int    nx = 0;
    int    ny = 0;
    // All buffers are N*N cell-centered. qx_*/qy_* follow the
    // "value lives on the right (resp. top) face of cell (i, j)" convention
    // used by both the wave decomposition and the SD shallow-water solver
    // (matches Sim2D.cu). No staggered scatter is required by downstream code.
    float* d_h_bar    = nullptr;
    float* d_h_tilde  = nullptr;
    float* d_qx_bar   = nullptr;
    float* d_qx_tilde = nullptr;
    float* d_qy_bar   = nullptr;
    float* d_qy_tilde = nullptr;
};

WaveDecompGpuPtrs waveDecomposeGpuDeviceOnly(const Grid& g, float d_grad_penalty, int n_diffusion_iters);
