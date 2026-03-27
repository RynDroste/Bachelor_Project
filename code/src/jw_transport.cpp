#include "jw_transport.h"

#include "jw_transport_gpu.hpp"

void jwTransportSurface(WaveDecomposition& dec,
                       const Grid&       gBar0,
                       const Grid&       gBar1,
                       float               halfW,
                       float               halfD,
                       float               dt,
                       float               gamma) {
    jwTransportSurfaceGpu(dec, gBar0, gBar1, halfW, halfD, dt, gamma);
}
