#pragma once

struct Grid;
struct WaveDecomposition;

void jwTransportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma);
