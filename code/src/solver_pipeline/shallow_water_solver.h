#pragma once

#include <vector>

// All fields live on a cell-centered NX*NY grid. qx and qy follow the
// "right-face / top-face owned by cell (i, j)" convention (matches Sim2D.cu):
//   QX(i, j) is the momentum at the right face of cell (i, j), shared with
//   cell (i+1, j); QY(i, j) is the momentum at the top face shared with
//   cell (i, j+1). The right face of cell (NX-1, j) is the right wall and is
//   pinned to 0 by the boundary kernels; the left wall is structurally absent
//   (the left face of cell (0, j) is implicit and treated as a closed wall).
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

// Shift the SWE state buffers by (di, dj) cells (integer grid units).
// Semantics: the SWE domain centre is translated by (+di*dx, +dj*dx) in world space,
// so the water column that used to live at local cell (i, j) now lives at (i - di, j - dj).
// Cells that become "new" (i.e. were previously outside the domain) are filled with the rest state:
//   H := restH, terrain := 0, Qx := 0, Qy := 0.
// This makes the SWE act as a moving window around a target (e.g. the player boat).
void gridSlideDomain(Grid& g, int di, int dj, float restH);

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
