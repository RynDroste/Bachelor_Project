// Jeschke & Wojtan 2023 Sec. 4.4 — advect surface tilde quantities with bulk velocity (semi-Lagrangian Alg. 3–4)

#pragma once

struct Grid;
struct WaveDecomposition;

// gBar0 / gBar1: bulk grid before / after SWE within the same step (same bathymetry)
void jwTransportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma);
