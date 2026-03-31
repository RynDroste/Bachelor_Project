#pragma once

struct Grid;
struct WaveDecomposition;

// Bar state before/after SWE in the same step.
void transportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma);
