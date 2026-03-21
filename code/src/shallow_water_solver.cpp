#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <vector>

// =============================================================================
// Constants
// =============================================================================

static constexpr float G          = 9.81f;
static constexpr float DRY_EPS    = 1e-4f;   // depth below which a cell is dry
static constexpr float CFL_FACTOR = 4.0f;    // |u|_max = dx / (CFL_FACTOR * dt)

// =============================================================================
// Grid
// =============================================================================

Grid::Grid(int nx, int ny, float cell_size, float timestep)
    : NX(nx), NY(ny), dx(cell_size), dt(timestep)
    , h       (nx * ny,        0.f)
    , qx      ((nx+1) * ny,    0.f)
    , qy      (nx * (ny+1),    0.f)
    , terrain (nx * ny,        0.f)
{}

float& Grid::H(int i, int j)       { return h[i + j*NX]; }
float  Grid::H(int i, int j) const { return h[i + j*NX]; }

float& Grid::QX(int i, int j)       { return qx[i + j*(NX+1)]; }
float  Grid::QX(int i, int j) const { return qx[i + j*(NX+1)]; }

float& Grid::QY(int i, int j)       { return qy[i + j*NX]; }
float  Grid::QY(int i, int j) const { return qy[i + j*NX]; }

float& Grid::B(int i, int j)       { return terrain[i + j*NX]; }
float  Grid::B(int i, int j) const { return terrain[i + j*NX]; }

// =============================================================================
// Helpers
// =============================================================================

// Upwind depth at x-face (i,j). Positive flow goes left->right.
static inline float faceH_X(const Grid& g, int i, int j) {
    int il = std::max(0, i-1);
    int ir = std::min(g.NX-1, i);
    return (g.QX(i,j) >= 0.f) ? g.H(il,j) : g.H(ir,j);
}

// Upwind depth at y-face (i,j). Positive flow goes down->up.
static inline float faceH_Y(const Grid& g, int i, int j) {
    int jd = std::max(0, j-1);
    int ju = std::min(g.NY-1, j);
    return (g.QY(i,j) >= 0.f) ? g.H(i,jd) : g.H(i,ju);
}

// Velocity at x-face (safe divide)
static inline float uX(const Grid& g, int i, int j) {
    float hf = faceH_X(g, i, j);
    return (hf < DRY_EPS) ? 0.f : g.QX(i,j) / hf;
}

// Velocity at y-face (safe divide)
static inline float uY(const Grid& g, int i, int j) {
    float hf = faceH_Y(g, i, j);
    return (hf < DRY_EPS) ? 0.f : g.QY(i,j) / hf;
}

// Average QX over two x-faces to get cell-center value
//   q→_avg[i,j] = (QX[i,j] + QX[i+1,j]) / 2
static inline float avgQX(const Grid& g, int i, int j) {
    return 0.5f * (g.QX(i,j) + g.QX(i+1,j));
}

// Average QY over two y-faces to get cell-center value
//   q↑_avg[i,j] = (QY[i,j] + QY[i,j+1]) / 2
static inline float avgQY(const Grid& g, int i, int j) {
    return 0.5f * (g.QY(i,j) + g.QY(i,j+1));
}

// =============================================================================
// Boundary conditions  (Section 4.6 of paper)
//   - Domain edges: reflecting (q = 0)
//   - Terrain-blocked faces: q = 0
// =============================================================================
static void applyBC(Grid& g) {
    // Domain edges
    for (int j = 0; j < g.NY; ++j) {
        g.QX(0,    j) = 0.f;
        g.QX(g.NX, j) = 0.f;
    }
    for (int i = 0; i < g.NX; ++i) {
        g.QY(i, 0   ) = 0.f;
        g.QY(i, g.NY) = 0.f;
    }

    // Terrain-blocked interior x-faces
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 1; i < g.NX; ++i) {
            float bmax  = std::max(g.B(i-1,j), g.B(i,j));
            float wL    = g.B(i-1,j) + g.H(i-1,j);
            float wR    = g.B(i,  j) + g.H(i,  j);
            if (bmax >= std::min(wL, wR) - DRY_EPS)
                g.QX(i, j) = 0.f;
        }
    }

    // Terrain-blocked interior y-faces
    for (int j = 1; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            float bmax  = std::max(g.B(i,j-1), g.B(i,j));
            float wD    = g.B(i,j-1) + g.H(i,j-1);
            float wU    = g.B(i,j  ) + g.H(i,j  );
            if (bmax >= std::min(wD, wU) - DRY_EPS)
                g.QY(i, j) = 0.f;
        }
    }
}

// =============================================================================
// Continuity equation — update h  (Eq. 18)
//
//   dh[i,j]/dt + (h[i+½,j]*u→[i+½,j] - h[i-½,j]*u→[i-½,j]) / dx
//              + (h[i,j+½]*u↑[i,j+½] - h[i,j-½]*u↑[i,j-½]) / dx = 0
//
//   h advances from t+dt/2 to t+3dt/2.
// =============================================================================
static void stepHeight(const Grid& g, std::vector<float>& h_out) {
    h_out.resize(g.NX * g.NY);
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            float hR = faceH_X(g, i+1, j);  float uR = uX(g, i+1, j);
            float hL = faceH_X(g, i,   j);  float uL = uX(g, i,   j);
            float hT = faceH_Y(g, i, j+1);  float vT = uY(g, i, j+1);
            float hB = faceH_Y(g, i, j  );  float vB = uY(g, i, j  );

            float divQ = (hR*uR - hL*uL + hT*vT - hB*vB) / g.dx;
            h_out[i + j*g.NX] = std::max(0.f, g.H(i,j) - g.dt * divQ);
        }
    }
}

// =============================================================================
// x-Momentum equation — update QX  (Eq. 19)
//
//   du→[i+½,j]/dt
//     + (q→_avg[i,j]   / h[i+½,j]) * (u→[i+½,j] - u→[i-½,j]) / dx    [x-adv]
//     + (q↑_mid[i,j-½] / h[i+½,j]) * (u→[i+½,j] - u→[i+½,j-1]) / dx  [y-adv]
//     + g * (h[i,j] - h[i-1,j]) / dx = 0
// =============================================================================
static void stepQX(const Grid& g, std::vector<float>& qx_out) {
    qx_out = g.qx;
    const float umax = g.dx / (CFL_FACTOR * g.dt);

    for (int j = 0; j < g.NY; ++j) {
        for (int i = 1; i < g.NX; ++i) {   // interior x-faces
            float hf = faceH_X(g, i, j);
            if (hf < DRY_EPS) { qx_out[i + j*(g.NX+1)] = 0.f; continue; }

            float u = uX(g, i, j);

            // x-advection: upwind on u→ using cell-center q→_avg
            float qx_avg = avgQX(g, i-1, j);   // avg at left cell (i-1,j)
            float du_dx;
            if (qx_avg >= 0.f) {
                du_dx = (u - uX(g, i-1, j)) / g.dx;
            } else {
                du_dx = (uX(g, i+1, j) - u) / g.dx;
            }

            // y-advection: q↑ averaged over the two y-faces bordering this x-face
            // q↑_mid = (QY[i-1, j] + QY[i, j]) / 2  (at the x-face row)
            float qy_mid = 0.5f * (
                g.QY(std::max(0,      i-1), j) +
                g.QY(std::min(g.NX-1, i  ), j)
            );
            float du_dy;
            if (qy_mid >= 0.f) {
                float u_d = (j > 0)      ? uX(g, i, j-1) : u;
                du_dy = (u - u_d) / g.dx;
            } else {
                float u_u = (j < g.NY-1) ? uX(g, i, j+1) : u;
                du_dy = (u_u - u) / g.dx;
            }

            // Pressure gradient
            float dh = g.H(std::min(g.NX-1,i), j) - g.H(std::max(0,i-1), j);
            float pres = G * dh / g.dx;

            float u_new = u - g.dt * ((qx_avg/hf)*du_dx + (qy_mid/hf)*du_dy + pres);
            u_new = std::max(-umax, std::min(umax, u_new));
            qx_out[i + j*(g.NX+1)] = u_new * hf;
        }
    }
}

// =============================================================================
// y-Momentum equation — update QY  (Eq. 20)
//
//   du↑[i,j+½]/dt
//     + (q→_mid[i-½,j] / h[i,j+½]) * (u↑[i,j+½] - u↑[i-1,j+½]) / dx  [x-adv]
//     + (q↑_avg[i,j]   / h[i,j+½]) * (u↑[i,j+½] - u↑[i,j-½]) / dx    [y-adv]
//     + g * (h[i,j] - h[i,j-1]) / dx = 0
// =============================================================================
static void stepQY(const Grid& g, std::vector<float>& qy_out) {
    qy_out = g.qy;
    const float vmax = g.dx / (CFL_FACTOR * g.dt);

    for (int j = 1; j < g.NY; ++j) {   // interior y-faces
        for (int i = 0; i < g.NX; ++i) {
            float hf = faceH_Y(g, i, j);
            if (hf < DRY_EPS) { qy_out[i + j*g.NX] = 0.f; continue; }

            float v = uY(g, i, j);

            // x-advection: q→ averaged over the two x-faces bordering this y-face
            float qx_mid = 0.5f * (
                g.QX(i, std::max(0,      j-1)) +
                g.QX(i, std::min(g.NY-1, j  ))
            );
            float dv_dx;
            if (qx_mid >= 0.f) {
                float v_l = (i > 0)      ? uY(g, i-1, j) : v;
                dv_dx = (v - v_l) / g.dx;
            } else {
                float v_r = (i < g.NX-1) ? uY(g, i+1, j) : v;
                dv_dx = (v_r - v) / g.dx;
            }

            // y-advection: upwind on u↑ using cell-center q↑_avg
            float qy_avg = avgQY(g, i, j-1);
            float dv_dy;
            if (qy_avg >= 0.f) {
                float v_d = (j > 1)    ? uY(g, i, j-1) : v;
                dv_dy = (v - v_d) / g.dx;
            } else {
                float v_u = (j < g.NY) ? uY(g, i, j+1) : v;
                dv_dy = (v_u - v) / g.dx;
            }

            // Pressure gradient
            float dh = g.H(i, std::min(g.NY-1,j)) - g.H(i, std::max(0,j-1));
            float pres = G * dh / g.dx;

            float v_new = v - g.dt * ((qx_mid/hf)*dv_dx + (qy_avg/hf)*dv_dy + pres);
            v_new = std::max(-vmax, std::min(vmax, v_new));
            qy_out[i + j*g.NX] = v_new * hf;
        }
    }
}

// =============================================================================
// Full SWE time step  (Section 4.2)
//   Input:  qx, qy at time t      h at time t+dt/2
//   Output: qx, qy at time t+dt   h at time t+3dt/2
//
// Continuity must use the updated face fluxes q^{n+1}; using old q^n here
// breaks the leapfrog coupling and leads to blow-up (spikes / NaN).
// =============================================================================
void sweStep(Grid& g) {
    applyBC(g);

    std::vector<float> qx_new, qy_new, h_new;
    stepQX(g, qx_new);
    stepQY(g, qy_new);

    g.qx.swap(qx_new);
    g.qy.swap(qy_new);
    applyBC(g);

    stepHeight(g, h_new);

    g.h = h_new;

    applyBC(g);
}