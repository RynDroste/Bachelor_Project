// One coupled substep: decompose, SWE, Airy, transport, recombine.

#pragma once

#include <vector>

struct Grid;
struct WaveDecomposition;
class AiryEWaveFFTW;

// `dec` is unused on the GPU-fused path; `hTildeSym` / `hTildePrevHalf` are not filled (symmetrization is on device).
void coupledSubstep(Grid& g, float halfW, float halfD,
                      WaveDecomposition& dec,
                      AiryEWaveFFTW& airy,
                      std::vector<float>& hTildeSym,
                      std::vector<float>& hTildePrevHalf,
                      bool& haveHtildePrevHalf,
                      float gradPenaltyD,
                      float transportGamma = 0.25f,
                      int waveDiffuseIters = 128);
