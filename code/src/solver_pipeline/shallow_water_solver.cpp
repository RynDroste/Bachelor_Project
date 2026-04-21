#include "solver_pipeline/shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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

void gridSlideDomain(Grid& g, int di, int dj, float restH) {
    if (di == 0 && dj == 0)
        return;

    const int NX = g.NX;
    const int NY = g.NY;

    std::vector<float> newH(static_cast<size_t>(NX) * NY, restH);
    std::vector<float> newT(static_cast<size_t>(NX) * NY, 0.0f);
    std::vector<float> newQx(static_cast<size_t>(NX + 1) * NY, 0.0f);
    std::vector<float> newQy(static_cast<size_t>(NX) * (NY + 1), 0.0f);

    // Cell-centred fields (H, terrain): layout NX x NY, indexed i + j*NX.
    {
        const int i0dst = std::max(0, -di);
        const int i1dst = std::min(NX, NX - di);
        const int srcI  = i0dst + di;
        const int len   = i1dst - i0dst;
        for (int j = 0; j < NY; ++j) {
            const int srcJ = j + dj;
            if (srcJ < 0 || srcJ >= NY)
                continue;
            if (len <= 0)
                continue;
            std::memcpy(&newH[i0dst + static_cast<size_t>(j) * NX],
                        &g.h[srcI + static_cast<size_t>(srcJ) * NX],
                        static_cast<size_t>(len) * sizeof(float));
            std::memcpy(&newT[i0dst + static_cast<size_t>(j) * NX],
                        &g.terrain[srcI + static_cast<size_t>(srcJ) * NX],
                        static_cast<size_t>(len) * sizeof(float));
        }
    }

    // Qx lives on vertical edges: layout (NX+1) x NY, indexed i + j*(NX+1), i in [0, NX].
    {
        const int stride = NX + 1;
        const int i0dst  = std::max(0, -di);
        const int i1dst  = std::min(stride, stride - di);
        const int srcI   = i0dst + di;
        const int len    = i1dst - i0dst;
        for (int j = 0; j < NY; ++j) {
            const int srcJ = j + dj;
            if (srcJ < 0 || srcJ >= NY)
                continue;
            if (len <= 0)
                continue;
            std::memcpy(&newQx[i0dst + static_cast<size_t>(j) * stride],
                        &g.qx[srcI + static_cast<size_t>(srcJ) * stride],
                        static_cast<size_t>(len) * sizeof(float));
        }
    }

    // Qy lives on horizontal edges: layout NX x (NY+1), indexed i + j*NX, j in [0, NY].
    {
        const int i0dst = std::max(0, -di);
        const int i1dst = std::min(NX, NX - di);
        const int srcI  = i0dst + di;
        const int len   = i1dst - i0dst;
        for (int j = 0; j <= NY; ++j) {
            const int srcJ = j + dj;
            if (srcJ < 0 || srcJ > NY)
                continue;
            if (len <= 0)
                continue;
            std::memcpy(&newQy[i0dst + static_cast<size_t>(j) * NX],
                        &g.qy[srcI + static_cast<size_t>(srcJ) * NX],
                        static_cast<size_t>(len) * sizeof(float));
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