// Jeschke & Wojtan 2023 §4.4 — 表面 tilde 量随 bulk 速度输运 (Alg. 3–4 的简化半拉格朗日版)

#pragma once

struct Grid;
struct WaveDecomposition;

// gBar0 / gBar1：同一步内 bulk 在 SWE 前 / 后的网格（地形一致）
void jwTransportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma);
