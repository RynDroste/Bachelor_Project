// Jeschke & Wojtan 2023 — Algorithm 1 单步子步：分解 → bulk(SWE) → surface(Airy) → 输运 → 合成

#pragma once

#include <vector>

struct Grid;
struct WaveDecomposition;
class AiryEWaveFFTW;

void jwCoupledSubstep(Grid& g, float halfW, float halfD,
                      WaveDecomposition& dec,
                      AiryEWaveFFTW& airy,
                      std::vector<float>& hTildeSym,
                      std::vector<float>& hTildePrevHalf,
                      bool& haveHtildePrevHalf,
                      float gradPenaltyD,
                      float transportGamma = 0.25f);
