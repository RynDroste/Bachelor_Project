#include "solver_pipeline/shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>

Grid::Grid(int nx, int ny, float cell_size, float timestep)
    : NX(nx), NY(ny), dx(cell_size), dt(timestep)
    , h       (nx * ny, 0.f)
    , qx      (nx * ny, 0.f)
    , qy      (nx * ny, 0.f)
    , terrain (nx * ny, 0.f)
{}

float& Grid::H(int i, int j)       { return h[i + j*NX]; }
float  Grid::H(int i, int j) const { return h[i + j*NX]; }

float& Grid::QX(int i, int j)       { return qx[i + j*NX]; }
float  Grid::QX(int i, int j) const { return qx[i + j*NX]; }

float& Grid::QY(int i, int j)       { return qy[i + j*NX]; }
float  Grid::QY(int i, int j) const { return qy[i + j*NX]; }

float& Grid::B(int i, int j)       { return terrain[i + j*NX]; }
float  Grid::B(int i, int j) const { return terrain[i + j*NX]; }

void gridSlideDomain(Grid& g, int di, int dj, float restH) {
    if (di == 0 && dj == 0)
        return;

    const int NX = g.NX;
    const int NY = g.NY;

    std::vector<float> newH(static_cast<size_t>(NX) * NY, restH);
    std::vector<float> newT(static_cast<size_t>(NX) * NY, 0.0f);
    std::vector<float> newQx(static_cast<size_t>(NX) * NY, 0.0f);
    std::vector<float> newQy(static_cast<size_t>(NX) * NY, 0.0f);

    // All four fields share the same cell-centered NX x NY layout, indexed i + j*NX.
    const int i0dst = std::max(0, -di);
    const int i1dst = std::min(NX, NX - di);
    const int srcI  = i0dst + di;
    const int len   = i1dst - i0dst;
    if (len > 0) {
        for (int j = 0; j < NY; ++j) {
            const int srcJ = j + dj;
            if (srcJ < 0 || srcJ >= NY)
                continue;
            const size_t dstOff = static_cast<size_t>(i0dst) + static_cast<size_t>(j) * NX;
            const size_t srcOff = static_cast<size_t>(srcI)  + static_cast<size_t>(srcJ) * NX;
            const size_t bytes  = static_cast<size_t>(len) * sizeof(float);
            std::memcpy(&newH[dstOff],  &g.h[srcOff],       bytes);
            std::memcpy(&newT[dstOff],  &g.terrain[srcOff], bytes);
            std::memcpy(&newQx[dstOff], &g.qx[srcOff],      bytes);
            std::memcpy(&newQy[dstOff], &g.qy[srcOff],      bytes);
        }
    }

    g.h       = std::move(newH);
    g.qx      = std::move(newQx);
    g.qy      = std::move(newQy);
    g.terrain = std::move(newT);
}

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
            // QX(i, j) is the right face of cell (i, j); the left face is the
            // right face of cell (i-1, j). At domain boundaries the missing
            // face is the closed wall (flux = 0).
            const float qxL = (i > 0)         ? g.QX(i - 1, j) : 0.f;
            const float qxR = (i < g.NX - 1)  ? g.QX(i, j)     : 0.f;
            const float qyB = (j > 0)         ? g.QY(i, j - 1) : 0.f;
            const float qyT = (j < g.NY - 1)  ? g.QY(i, j)     : 0.f;
            const float hu    = 0.5f * (qxL + qxR);
            const float hv    = 0.5f * (qyB + qyT);
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
