#include "transport.h"

#include "transport_gpu.hpp"

void transportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma) {
    transportSurfaceGpu(dec, gBar0, gBar1, halfW, halfD, dt, gamma);
}
