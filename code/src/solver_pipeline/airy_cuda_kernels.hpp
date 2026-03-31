#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

void airy_cuda_launch_real_to_complex(const float* d_re, cufftComplex* d_out, int n, cudaStream_t stream = 0);

void airy_cuda_launch_spectral_qx(const cufftComplex* hatH,
                                    const cufftComplex* hatQx,
                                    cufftComplex* out,
                                    int nx,
                                    int ny,
                                    float dx,
                                    float dy,
                                    float dt,
                                    float g,
                                    float hDepth,
                                    cudaStream_t stream = 0);

void airy_cuda_launch_spectral_qy(const cufftComplex* hatH,
                                    const cufftComplex* hatQy,
                                    cufftComplex* out,
                                    int nx,
                                    int ny,
                                    float dx,
                                    float dy,
                                    float dt,
                                    float g,
                                    float hDepth,
                                    cudaStream_t stream = 0);

void airy_cuda_launch_cpx_to_real_scaled(const cufftComplex* d_in,
                                         float* d_out,
                                         float scale,
                                         int n,
                                         cudaStream_t stream = 0);

void airy_cuda_launch_blend(const float* d_sx0,
                            const float* d_sx1,
                            const float* d_sx2,
                            const float* d_sx3,
                            const float* d_sy0,
                            const float* d_sy1,
                            const float* d_sy2,
                            const float* d_sy3,
                            const float* d_h_bar,
                            float* d_qxc,
                            float* d_qyc,
                            int nx,
                            int ny,
                            cudaStream_t stream = 0);

void airy_cuda_launch_face_to_cell(const float* d_qx_face,
                                   const float* d_qy_face,
                                   float* d_qxc,
                                   float* d_qyc,
                                   int nx,
                                   int ny,
                                   cudaStream_t stream = 0);

void airy_cuda_launch_cell_to_qx_faces(const float* d_qxc, float* d_qx_face, int nx, int ny, cudaStream_t stream = 0);

void airy_cuda_launch_cell_to_qy_faces(const float* d_qyc, float* d_qy_face, int nx, int ny, cudaStream_t stream = 0);
