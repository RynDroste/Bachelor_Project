#pragma once

#include <vector>

struct Grid {
    int   NX, NY;
    float dx;   // uniform cell width (m)
    float dt;   // time step (s)

    std::vector<float> h;         // [NX * NY]
    std::vector<float> qx;        // [(NX+1) * NY]
    std::vector<float> qy;        // [NX * (NY+1)]
    std::vector<float> terrain;   // [NX * NY] static bed elevation

    Grid(int nx, int ny, float cell_size, float timestep);

    float& H(int i, int j);
    float  H(int i, int j) const;

    // qx face (i,j): between cell (i-1,j) and (i,j), i in [0,NX]
    float& QX(int i, int j);
    float  QX(int i, int j) const;

    // qy face (i,j): between cell (i,j-1) and (i,j), j in [0,NY]
    float& QY(int i, int j);
    float  QY(int i, int j) const;

    float& B(int i, int j);
    float  B(int i, int j) const;
};

// GPU SWE step.
void sweStepGpu(Grid& g);

// GPU boundary conditions and wet/dry handling.
void sweApplyBoundaryConditionsGpu(Grid& g);

// Coupled pipeline (device-resident): ensure SWE d_terrain matches g (cheap if cached).
void swePrefetchDeviceTerrain(const Grid& g);

// Run one SWE step on device buffers (in/out). Uses staging through internal SWE pool.
void sweStepGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny, float dx,
                             float dt);

void sweApplyBoundaryGpuInPlaceDevice(float* d_h, float* d_qx, float* d_qy, const Grid& gTerrainRef, int nx, int ny,
                                      float dx, float dt);

void sweDownloadGridFromDevice(Grid& g, const float* d_h, const float* d_qx, const float* d_qy);

// Cell-centered |u|, Fr = |u|/sqrt(g*h) from averaged face fluxes; only cells with h > dryEps.
struct ShallowWaterDiagnostics {
    float fr_max;           // max Froude number
    float speed_at_fr_max;  // |u| at cell where Fr is maximal
    float h_at_fr_max;      // h at that cell
    float speed_max;        // max |u| over wet cells
    float h_min_wet;        // min h over wet cells (0 if none)
};

ShallowWaterDiagnostics gridShallowWaterDiagnostics(const Grid& g, float gravity = 9.81f, float dryEps = 1e-3f);
