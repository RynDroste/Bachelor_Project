#include "jw_pipeline.h"

#include "airy_fftw.h"
#include "jw_transport.h"
#include "shallow_water_solver.h"
#include "wavedecomposer.h"

#include <algorithm>
#include <memory>

namespace {

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

void assignBarState(Grid& dst, const Grid& terrainSrc, const WaveDecomposition& dec) {
    for (int j = 0; j < dst.NY; ++j) {
        for (int i = 0; i < dst.NX; ++i) {
            dst.B(i, j) = terrainSrc.B(i, j);
            dst.H(i, j) = dec.h_bar[i + j * dst.NX];
        }
    }
    for (int j = 0; j < dst.NY; ++j) {
        for (int i = 0; i <= dst.NX; ++i)
            dst.QX(i, j) = dec.qx_bar[i + j * (dst.NX + 1)];
    }
    for (int j = 0; j <= dst.NY; ++j) {
        for (int i = 0; i < dst.NX; ++i)
            dst.QY(i, j) = dec.qy_bar[i + j * dst.NX];
    }
}

} // namespace

void jwCoupledSubstep(Grid& g, float halfW, float halfD,
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
    sweStep(gBar1);

    const size_t nh = dec.h_tilde.size();
    if (hTildeSym.size() != nh) {
        hTildeSym.resize(nh);
        hTildePrevHalf.resize(nh);
    }
    if (!haveHtildePrevHalf)
        std::copy(dec.h_tilde.begin(), dec.h_tilde.end(), hTildeSym.begin());
    else {
        for (size_t i = 0; i < nh; ++i)
            hTildeSym[i] = 0.5f * (hTildePrevHalf[i] + dec.h_tilde[i]);
    }

    airy.step(dt, 9.81f, hTildeSym.data(), dec.h_bar.data(), dec.qx_tilde.data(), dec.qy_tilde.data());

    jwTransportSurface(dec, gBar0, gBar1, halfW, halfD, dt, transportGamma);

    std::copy(dec.h_tilde.begin(), dec.h_tilde.end(), hTildePrevHalf.begin());
    haveHtildePrevHalf = true;

    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i)
            g.H(i, j) = gBar1.H(i, j) + dec.h_tilde[i + j * g.NX];
    }
    for (int j = 0; j < g.NY; ++j) {
        for (int i = 0; i <= g.NX; ++i)
            g.QX(i, j) = gBar1.QX(i, j) + dec.qx_tilde[i + j * (g.NX + 1)];
    }
    for (int j = 0; j <= g.NY; ++j) {
        for (int i = 0; i < g.NX; ++i)
            g.QY(i, j) = gBar1.QY(i, j) + dec.qy_tilde[i + j * g.NX];
    }

    sweApplyBoundaryConditions(g);
}
