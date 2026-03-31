#include "solver_pipeline/shallow_water_solver.h"

#include <cmath>

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

ShallowWaterDiagnostics gridShallowWaterDiagnostics(const Grid& g, float gravity, float dryEps) {
    ShallowWaterDiagnostics d{};
    float    hMinWet = 1e30f;
    unsigned wetCount = 0;
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i) {
            const float h = g.H(i, j);
            if (h < dryEps)
                continue;
            ++wetCount;
            if (h < hMinWet)
                hMinWet = h;
            const float hu    = 0.5f * (g.QX(i, j) + g.QX(i + 1, j));
            const float hv    = 0.5f * (g.QY(i, j) + g.QY(i, j + 1));
            const float u     = hu / h;
            const float v     = hv / h;
            const float speed = std::sqrt(u * u + v * v);
            if (speed > d.speed_max)
                d.speed_max = speed;
            const float c = std::sqrt(gravity * h);
            if (c > 1e-12f) {
                const float fr = speed / c;
                if (fr > d.fr_max) {
                    d.fr_max           = fr;
                    d.speed_at_fr_max  = speed;
                    d.h_at_fr_max      = h;
                }
            }
        }
    }
    d.h_min_wet = (wetCount > 0) ? hMinWet : 0.f;
    return d;
}