#include "solver_pipeline/pipeline.h"

#include "solver_pipeline/airy_fftw.h"
#include "solver_pipeline/transport.h"
#include "solver_pipeline/shallow_water_solver.h"
#include "solver_pipeline/wavedecomposer.h"

#include <algorithm>
#include <cstring>
#include <memory>

namespace {

// Reuse bar grids across substeps.
struct BarGridScratch {
    std::unique_ptr<Grid> gBar0;
    std::unique_ptr<Grid> gBar1;
    void ensureLike(const Grid& g) {
        if (!gBar0 || gBar0->NX != g.NX || gBar0->NY != g.NY || gBar0->dx != g.dx || gBar0->dt != g.dt) {
            gBar0 = std::make_unique<Grid>(g.NX, g.NY, g.dx, g.dt);
            gBar1 = std::make_unique<Grid>(g.NX, g.NY, g.dx, g.dt);
        }
    }
};
static BarGridScratch g_barScratch;

// dec.* and Grid fields share the same row-major layouts (see Grid::H/QX/QY and WaveDecomposition).
void assignBarState(Grid& dst, const Grid& terrainSrc, const WaveDecomposition& dec) {
    const int            nx    = dst.NX;
    const int            ny    = dst.NY;
    const size_t         ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t         nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    const size_t         nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);
    const size_t         bTerrain = ncell * sizeof(float);

    std::memcpy(dst.terrain.data(), terrainSrc.terrain.data(), bTerrain);
    std::memcpy(dst.h.data(), dec.h_bar.data(), bTerrain);
    std::memcpy(dst.qx.data(), dec.qx_bar.data(), nqx * sizeof(float));
    std::memcpy(dst.qy.data(), dec.qy_bar.data(), nqy * sizeof(float));
}

void recombineBarPlusTilde(Grid& g, const Grid& gBar1, const WaveDecomposition& dec) {
    const int    nx    = g.NX;
    const int    ny    = g.NY;
    const size_t ncell = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t nqx   = static_cast<size_t>(nx + 1) * static_cast<size_t>(ny);
    const size_t nqy   = static_cast<size_t>(nx) * static_cast<size_t>(ny + 1);

    const float* hb  = gBar1.h.data();
    const float* ht  = dec.h_tilde.data();
    float*       gh  = g.h.data();
    for (size_t k = 0; k < ncell; ++k)
        gh[k] = hb[k] + ht[k];

    const float* qxb = gBar1.qx.data();
    const float* qxt = dec.qx_tilde.data();
    float*       gqx = g.qx.data();
    for (size_t k = 0; k < nqx; ++k)
        gqx[k] = qxb[k] + qxt[k];

    const float* qyb = gBar1.qy.data();
    const float* qyt = dec.qy_tilde.data();
    float*       gqy = g.qy.data();
    for (size_t k = 0; k < nqy; ++k)
        gqy[k] = qyb[k] + qyt[k];
}

} // namespace

void coupledSubstep(Grid& g, float halfW, float halfD,
                      WaveDecomposition& dec,
                      AiryEWaveFFTW& airy,
                      std::vector<float>& hTildeSym,
                      std::vector<float>& hTildePrevHalf,
                      bool& haveHtildePrevHalf,
                      float gradPenaltyD,
                      float transportGamma,
                      int waveDiffuseIters) {
    const float dt = g.dt;

    waveDecompose(g, gradPenaltyD, waveDiffuseIters, dec);

    g_barScratch.ensureLike(g);
    Grid& gBar0 = *g_barScratch.gBar0;
    Grid& gBar1 = *g_barScratch.gBar1;
    assignBarState(gBar0, g, dec);
    assignBarState(gBar1, g, dec);
    sweStepGpu(gBar1);

    const size_t nh = dec.h_tilde.size();
    if (hTildeSym.size() != nh) {
        hTildeSym.resize(nh);
        hTildePrevHalf.resize(nh);
    }
    if (!haveHtildePrevHalf)
        std::copy(dec.h_tilde.begin(), dec.h_tilde.end(), hTildeSym.begin());
    else {
        const float* prev = hTildePrevHalf.data();
        const float* cur  = dec.h_tilde.data();
        float*       out  = hTildeSym.data();
        for (size_t k = 0; k < nh; ++k)
            out[k] = 0.5f * (prev[k] + cur[k]);
    }

    airy.step(dt, 9.81f, hTildeSym.data(), dec.h_bar.data(), dec.qx_tilde.data(), dec.qy_tilde.data());

    transportSurface(dec, gBar0, gBar1, halfW, halfD, dt, transportGamma);

    std::copy(dec.h_tilde.begin(), dec.h_tilde.end(), hTildePrevHalf.begin());
    haveHtildePrevHalf = true;

    recombineBarPlusTilde(g, gBar1, dec);

    sweApplyBoundaryConditionsGpu(g);
}
