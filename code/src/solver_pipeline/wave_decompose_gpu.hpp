#pragma once

#include <cstddef>

struct Grid;
struct WaveDecomposition;

void waveDecomposeGpu(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out);

struct WaveDecompGpuPtrs {
    int    nx = 0;
    int    ny = 0;
    // All bar/tilde buffers are now N*N (cell-centered). qx_*/qy_* follow the
    // "value lives on the right (resp. top) face of cell (i,j)" convention used
    // internally by the wave decomposition (matches Sim2D.cu). Downstream code
    // that needs staggered (NX+1)*NY / NX*(NY+1) face arrays must scatter
    // explicitly with the helpers in pipeline_coupled_gpu.cu.
    float* d_h_bar    = nullptr;
    float* d_h_tilde  = nullptr;
    float* d_qx_bar   = nullptr;
    float* d_qx_tilde = nullptr;
    float* d_qy_bar   = nullptr;
    float* d_qy_tilde = nullptr;
};

WaveDecompGpuPtrs waveDecomposeGpuDeviceOnly(const Grid& g, float d_grad_penalty, int n_diffusion_iters);
