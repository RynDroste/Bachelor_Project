#pragma once

#include <vector>

struct Grid {
    int   NX, NY;
    float dx;
    float dt;

    std::vector<float> h;
    std::vector<float> qx;
    std::vector<float> qy;
    std::vector<float> terrain;

    Grid(int nx, int ny, float cell_size, float timestep);

    float& H(int i, int j);
    float  H(int i, int j) const;

    float& QX(int i, int j);
    float  QX(int i, int j) const;

    float& QY(int i, int j);
    float  QY(int i, int j) const;

    float& B(int i, int j);
    float  B(int i, int j) const;
};

void sweStepGpu(Grid& g);

void sweApplyBoundaryConditionsGpu(Grid& g);

void swePrefetchDeviceTerrain(const Grid& g);

void sweStepGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny, float dx,
                             float dt);

void sweApplyBoundaryGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny,
                                      float dx, float dt);

void sweDownloadGridFromDevice(Grid& g, const float* d_h, const float* d_qx, const float* d_qy);

struct ShallowWaterDiagnostics {
    float fr_max;
    float speed_at_fr_max;
    float h_at_fr_max;
    float speed_max;
    float h_min_wet;
};

ShallowWaterDiagnostics gridShallowWaterDiagnostics(const Grid& g, float gravity = 9.81f, float dryEps = 1e-3f);
