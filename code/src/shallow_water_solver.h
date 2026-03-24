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

    float& QX(int i, int j);
    float  QX(int i, int j) const;

    float& QY(int i, int j);
    float  QY(int i, int j) const;

    float& B(int i, int j);
    float  B(int i, int j) const;
};

void sweStep(Grid& g);

void sweApplyBoundaryConditions(Grid& g);
