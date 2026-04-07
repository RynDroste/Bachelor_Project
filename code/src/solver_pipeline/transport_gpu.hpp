#pragma once

struct Grid;
struct WaveDecomposition;

void transportSurfaceGpu(WaveDecomposition& dec,
                          const Grid&       gBar0,
                          const Grid&       gBar1,
                          float               halfW,
                          float               halfD,
                          float               dt,
                          float               gamma);

void transportSurfaceGpuDevice(float* d_bar0_h, float* d_bar0_qx, float* d_bar0_qy, float* d_bar1_h, float* d_bar1_qx,
                               float* d_bar1_qy, float* d_h_tilde, float* d_qx_tilde, float* d_qy_tilde, int nx, int ny,
                               float dx, float halfW, float halfD, float dt, float gamma);
