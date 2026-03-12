#pragma once

#include <vector>

class ShallowWaterSolver {
public:
    explicit ShallowWaterSolver(int gridSize);

    void advance(float frameDt);

    float etaAt(int i, int j) const;
    int resolution() const;

private:
    int idxEta(int i, int j) const;
    int idxU(int i, int jFace) const;
    int idxV(int iFace, int j) const;

    void step();

    int gridSize_;
    int N_;
    float dx_;
    float dy_;
    float targetDt_;
    float H_;
    float g_;
    float f_;
    float linearDrag_;
    float energyThreshold_;
    int lowEnergyStepsRequired_;
    float dt_;
    float dampingFactor_;

    std::vector<float> etaCurr_;
    std::vector<float> etaNext_;
    std::vector<float> uCurr_;
    std::vector<float> uNext_;
    std::vector<float> vCurr_;
    std::vector<float> vNext_;

    float accumulator_;
    int lowEnergySteps_;
    bool simulationActive_;
};
