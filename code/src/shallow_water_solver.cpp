#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <vector>

static constexpr float G          = 9.81f;
static constexpr float DRY_EPS    = 1e-4f;
static constexpr float CFL_FACTOR = 4.0f;

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

static inline float faceH_X(const Grid& g, int i, int j) {
    int il = std::max(0, i-1);
    int ir = std::min(g.NX-1, i);
    return (g.QX(i,j) >= 0.f) ? g.H(il,j) : g.H(ir,j);
}

static inline float faceH_Y(const Grid& g, int i, int j) {
    int jd = std::max(0, j-1);
    int ju = std::min(g.NY-1, j);
    return (g.QY(i,j) >= 0.f) ? g.H(i,jd) : g.H(i,ju);
}

static inline float uX(const Grid& g, int i, int j) {
    float hf = faceH_X(g, i, j);
    return (hf < DRY_EPS) ? 0.f : g.QX(i,j) / hf;
}

static inline float uY(const Grid& g, int i, int j) {
    float hf = faceH_Y(g, i, j);
    return (hf < DRY_EPS) ? 0.f : g.QY(i,j) / hf;
}

static inline float avgQX(const Grid& g, int i, int j) {
    return 0.5f * (g.QX(i,j) + g.QX(i+1,j));
}

static inline float avgQY(const Grid& g, int i, int j) {
    return 0.5f * (g.QY(i,j) + g.QY(i,j+1));
}

void sweApplyBoundaryConditions(Grid& g) {
    for (int j = 0; j < g.NY; ++j) {
        g.QX(0,    j) = 0.f;
        g.QX(g.NX, j) = 0.f;
    }
    for (int i = 0; i < g.NX; ++i) {
        g.QY(i, 0   ) = 0.f;
        g.QY(i, g.NY) = 0.f;
    }

    for (int j = 0; j < g.NY; ++j) {
        for (int i = 1; i < g.NX; ++i) {
            float bmax  = std::max(g.B(i-1,j), g.B(i,j));
            float wL    = g.B(i-1,j) + g.H(i-1,j);
            float wR    = g.B(i,  j) + g.H(i,  j);
            if (bmax >= std::min(wL, wR) - DRY_EPS)
                g.QX(i, j) = 0.f;
        }
    }

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

static void stepQX(const Grid& g, std::vector<float>& qx_out) {
    qx_out = g.qx;
    const float umax = g.dx / (CFL_FACTOR * g.dt);

    for (int j = 0; j < g.NY; ++j) {
        for (int i = 1; i < g.NX; ++i) {
            float hf = faceH_X(g, i, j);
            if (hf < DRY_EPS) { qx_out[i + j*(g.NX+1)] = 0.f; continue; }

            float u = uX(g, i, j);

            float qx_avg = avgQX(g, i-1, j);
            float du_dx;
            if (qx_avg >= 0.f) {
                du_dx = (u - uX(g, i-1, j)) / g.dx;
            } else {
                du_dx = (uX(g, i+1, j) - u) / g.dx;
            }

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

            const float hL = g.H(std::max(0, i - 1), j);
            const float hR = g.H(std::min(g.NX - 1, i), j);
            float pres = 0.f;
            if (hL >= DRY_EPS && hR >= DRY_EPS) {
                const float dh = hR - hL;
                pres = G * dh / g.dx;
            }

            const float advection = (qx_avg / hf) * du_dx + (qy_mid / hf) * du_dy;
            const float qx_current = g.QX(i, j);
            float qx_next = qx_current - g.dt * (hf * advection + hf * pres);
            const float qmax = hf * umax;
            qx_next = std::max(-qmax, std::min(qmax, qx_next));
            qx_out[i + j*(g.NX+1)] = qx_next;
        }
    }
}

static void stepQY(const Grid& g, std::vector<float>& qy_out) {
    qy_out = g.qy;
    const float vmax = g.dx / (CFL_FACTOR * g.dt);

    for (int j = 1; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            float hf = faceH_Y(g, i, j);
            if (hf < DRY_EPS) { qy_out[i + j*g.NX] = 0.f; continue; }

            float v = uY(g, i, j);

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

            float qy_avg = avgQY(g, i, j-1);
            float dv_dy;
            if (qy_avg >= 0.f) {
                float v_d = (j > 1)    ? uY(g, i, j-1) : v;
                dv_dy = (v - v_d) / g.dx;
            } else {
                float v_u = (j < g.NY) ? uY(g, i, j+1) : v;
                dv_dy = (v_u - v) / g.dx;
            }

            const float hD = g.H(i, std::max(0, j - 1));
            const float hU = g.H(i, std::min(g.NY - 1, j));
            float pres = 0.f;
            if (hD >= DRY_EPS && hU >= DRY_EPS) {
                const float dh = hU - hD;
                pres = G * dh / g.dx;
            }

            const float advection = (qx_mid / hf) * dv_dx + (qy_avg / hf) * dv_dy;
            const float qy_current = g.QY(i, j);
            float qy_next = qy_current - g.dt * (hf * advection + hf * pres);
            const float qmax = hf * vmax;
            qy_next = std::max(-qmax, std::min(qmax, qy_next));
            qy_out[i + j*g.NX] = qy_next;
        }
    }
}

void sweStep(Grid& g) {
    sweApplyBoundaryConditions(g);

    std::vector<float> qx_new, qy_new, h_new;
    stepQX(g, qx_new);
    stepQY(g, qy_new);

    g.qx.swap(qx_new);
    g.qy.swap(qy_new);
    sweApplyBoundaryConditions(g);

    stepHeight(g, h_new);

    g.h = h_new;

    sweApplyBoundaryConditions(g);
}