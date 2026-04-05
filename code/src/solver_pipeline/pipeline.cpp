#include "solver_pipeline/pipeline.h"

#include "solver_pipeline/pipeline_coupled_gpu.hpp"

void coupledSubstep(Grid& g, float halfW, float halfD, WaveDecomposition& dec, AiryEWaveFFTW& airy,
                    std::vector<float>& hTildeSym, std::vector<float>& hTildePrevHalf, bool& haveHtildePrevHalf,
                    float gradPenaltyD, float transportGamma, int waveDiffuseIters) {
    coupledSubstepGpu(g, halfW, halfD, dec, airy, hTildeSym, hTildePrevHalf, haveHtildePrevHalf, gradPenaltyD,
                      transportGamma, waveDiffuseIters);
}
