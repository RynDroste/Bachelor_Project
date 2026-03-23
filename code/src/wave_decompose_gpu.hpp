#pragma once

struct Grid;
struct WaveDecomposition;

void waveDecomposeGpu(const Grid& g, float d_grad_penalty, int n_diffusion_iters, WaveDecomposition& out);
