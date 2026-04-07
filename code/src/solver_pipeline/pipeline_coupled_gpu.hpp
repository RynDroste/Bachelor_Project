#pragma once

#include <vector>

struct Grid;
struct WaveDecomposition;
class AiryEWaveFFTW;

void coupledSubstepGpu(Grid& g, float halfW, float halfD, WaveDecomposition& dec, AiryEWaveFFTW& airy,
                       std::vector<float>& hTildeSym, std::vector<float>& hTildePrevHalf, bool& haveHtildePrevHalf,
                       float gradPenaltyD, float transportGamma, int waveDiffuseIters);
