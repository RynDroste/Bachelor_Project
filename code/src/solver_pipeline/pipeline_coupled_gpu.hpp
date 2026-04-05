#pragma once

#include <vector>

struct Grid;
struct WaveDecomposition;
class AiryEWaveFFTW;

// Fused device pipeline for one coupled substep (see pipeline.h).
void coupledSubstepGpu(Grid& g, float halfW, float halfD, WaveDecomposition& dec, AiryEWaveFFTW& airy,
                       std::vector<float>& hTildeSym, std::vector<float>& hTildePrevHalf, bool& haveHtildePrevHalf,
                       float gradPenaltyD, float transportGamma, int waveDiffuseIters);
