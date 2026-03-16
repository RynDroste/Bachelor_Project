#pragma once

#include <vector>

class PressureSolver {
public:
    PressureSolver(int resolution, float dx, float dy);

    void setDensity(float rhoValue);
    void setMeanDepth(float depthValue);
    void setNonHydrostaticStrength(float strengthValue);
    void setGravity(float gValue);
    void setIterations(int maxIters);
    void setTolerance(float eps);

    float project(
        std::vector<float>& etaField,
        std::vector<float>& uField,
        std::vector<float>& vField,
        float dt
    );

    const std::vector<float>& pressure() const;

private:
    int idxCell(int i, int j) const;
    int idxU(int i, int jFace) const;
    int idxV(int iFace, int j) const;

    void buildRhs(
        const std::vector<float>& etaField,
        const std::vector<float>& uField,
        const std::vector<float>& vField,
        float dt
    );
    void solveCoupledPressure(float dt);
    void applyPressureGradient(
        std::vector<float>& uField,
        std::vector<float>& vField,
        float dt
    ) const;
    void applyEtaCorrection(
        std::vector<float>& etaField,
        const std::vector<float>& uField,
        const std::vector<float>& vField,
        float dt
    ) const;
    float computeMaxAbsDivergence(
        const std::vector<float>& uField,
        const std::vector<float>& vField
    ) const;
    float divergenceAtCell(const std::vector<float>& uField, const std::vector<float>& vField, int i, int j) const;
    float gradXAtUFace(const std::vector<float>& cellField, int i, int jFace) const;
    float gradYAtVFace(const std::vector<float>& cellField, int iFace, int j) const;
    float applyCoupledOperatorAtCell(const std::vector<float>& qField, float dt, int i, int j) const;
    void applyCoupledOperator(const std::vector<float>& in, std::vector<float>& out, float dt) const;
    float dotCells(const std::vector<float>& a, const std::vector<float>& b) const;

    int N;
    float dx;
    float dy;
    float rho;
    float meanDepth;
    float nhStrength;
    float g;
    int maxIterations;
    float tolerance;

    std::vector<float> rhs;
    std::vector<float> p;
    std::vector<float> pNext;
    std::vector<float> cgR;
    std::vector<float> cgZ;
    std::vector<float> cgP;
    std::vector<float> cgAp;
};