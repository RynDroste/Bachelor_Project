#pragma once

#include <vector>

class PressureSolver {
public:
    PressureSolver(int resolution, float dx, float dy);

    void setDensity(float rhoValue);
    void setMeanDepth(float depthValue);
    void setIterations(int maxIters);
    void setTolerance(float eps);

    float project(std::vector<float>& uField, std::vector<float>& vField, float dt);

    const std::vector<float>& pressure() const;

private:
    int idxCell(int i, int j) const;
    int idxU(int i, int jFace) const;
    int idxV(int iFace, int j) const;

    void buildRhs(
        const std::vector<float>& uField,
        const std::vector<float>& vField,
        float dt
    );
    void solvePoisson();
    void applyPressureGradient(
        std::vector<float>& uField,
        std::vector<float>& vField,
        float dt
    ) const;
    float computeMaxAbsDivergence(
        const std::vector<float>& uField,
        const std::vector<float>& vField
    ) const;

    int N;
    float dx;
    float dy;
    float rho;
    float meanDepth;
    int maxIterations;
    float tolerance;

    std::vector<float> rhs;
    std::vector<float> p;
    std::vector<float> pNext;
};