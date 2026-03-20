#pragma once

#include <vector>

#include "pressure_solver.h"

class ShallowWaterSolver {
public:
    explicit ShallowWaterSolver(int gridSize);
    ShallowWaterSolver(int gridSize, float dxMeters, float dyMeters);
    void setBathymetry(const std::vector<float>& bedElevation);
    void setInjection(int centerI, int centerJ, float radiusCells, float rate, float durationSeconds);

    void advance(float frameDt);

    float etaAt(int i, int j) const;
    int resolution() const;

private:
    int idxEta(int i, int j) const;
    int idxU(int i, int jFace) const;
    int idxV(int iFace, int j) const;
    float localDepth(const std::vector<float>& etaField, int i, int j) const;

    void step();
    void computeRhs(
        const std::vector<float>& etaField,
        const std::vector<float>& uField,
        const std::vector<float>& vField,
        std::vector<float>& etaRhs,
        std::vector<float>& uRhs,
        std::vector<float>& vRhs
    ) const;
    void enforceVelocityBoundaries(std::vector<float>& uField, std::vector<float>& vField) const;
    float sampleEta(const std::vector<float>& etaField, float xEta, float yEta) const;
    float sampleU(const std::vector<float>& uField, float xEta, float yEta) const;
    float sampleV(const std::vector<float>& vField, float xEta, float yEta) const;
    float bilinearSample(const std::vector<float>& field, int rows, int cols, float rowCoord, float colCoord) const;
    float spongeSigma(float distanceToBoundaryCells) const;
    void applyBoundarySponge(
        std::vector<float>& etaField,
        std::vector<float>& uField,
        std::vector<float>& vField,
        float dtStep
    ) const;
    void updateTimeStepFromCfl();
    void applyShapiroFilter(std::vector<float>& etaField) const;
    float injectionSourceAtCell(int i, int j) const;

    int N;
    float dx;
    float dy;
    float targetDt;
    float H;
    float g;
    float f;
    float linearDrag;
    float cflLimit;
    float shapiroStrength;
    int spongeWidthCells;
    float spongeMaxSigma;
    float energyThreshold;
    int lowEnergyStepsRequired;
    float dt;
    float dryDepthThreshold;

    std::vector<float> etaCurr;
    std::vector<float> etaNext;
    std::vector<float> etaStage;
    std::vector<float> etaRhs;
    std::vector<float> uCurr;
    std::vector<float> uNext;
    std::vector<float> uStage;
    std::vector<float> uRhs;
    std::vector<float> vCurr;
    std::vector<float> vNext;
    std::vector<float> vStage;
    std::vector<float> vRhs;
    std::vector<float> bathymetry;

    PressureSolver pressureSolver;
    bool enablePressureProjection;

    float accumulator;
    float simulationTime;
    int lowEnergySteps;
    bool simulationActive;
    bool injectionEnabled;
    int injectionCenterI;
    int injectionCenterJ;
    float injectionRadiusCells;
    float injectionRate;
    float injectionEndTime;
};
