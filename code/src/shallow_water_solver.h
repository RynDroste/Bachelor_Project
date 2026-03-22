// =============================================================================
// SWE Solver — Stelling & Duinmeijer 2003 (momentum-conserving scheme)
// Corresponding to Appendix A of Jeschke & Wojtan, SIGGRAPH 2023
//
// Staggered grid:
//   h[i,j]  — cell centers, water depth
//   qx[i,j] — x-faces between (i-1,j) and (i,j), i in [0,NX]
//   qy[i,j] — y-faces between (i,j-1) and (i,j), j in [0,NY]
//
// Leapfrog: h at half steps (t+dt/2), q at integer times t.
// =============================================================================

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

// Input:  qx, qy at t; h at t+dt/2
// Output: qx, qy at t+dt; h at t+3dt/2
void sweStep(Grid& g);

// Domain boundaries and wet/dry; apply again after J&W bar+tilde merge
void sweApplyBoundaryConditions(Grid& g);
