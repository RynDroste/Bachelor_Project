#include "pressure_solver.h"

#include <algorithm>
#include <cmath>

PressureSolver::PressureSolver(int resolution, float dx, float dy)
    : N(resolution),
      dx(dx),
      dy(dy),
      rho(1.0f),
      meanDepth(1.0f),
      nhStrength(1.0f),
      g(9.81f),
      maxIterations(60),
      tolerance(1e-5f),
      rhs(N * N, 0.0f),
      p(N * N, 0.0f),
      pNext(N * N, 0.0f),
      cgR(N * N, 0.0f),
      cgZ(N * N, 0.0f),
      cgP(N * N, 0.0f),
      cgAp(N * N, 0.0f) {}

void PressureSolver::setDensity(float rhoValue) {
    rho = std::max(rhoValue, 1e-6f);
}

void PressureSolver::setMeanDepth(float depthValue) {
    meanDepth = std::max(depthValue, 1e-4f);
}

void PressureSolver::setNonHydrostaticStrength(float strengthValue) {
    nhStrength = std::clamp(strengthValue, 0.0f, 1.0f);
}

void PressureSolver::setGravity(float gValue) {
    g = std::max(gValue, 1e-6f);
}

void PressureSolver::setIterations(int maxIters) {
    maxIterations = std::max(maxIters, 1);
}

void PressureSolver::setTolerance(float eps) {
    tolerance = std::max(eps, 1e-9f);
}

float PressureSolver::project(
    std::vector<float>& etaField,
    std::vector<float>& uField,
    std::vector<float>& vField,
    float dt
) {
    if (N <= 1 || dt <= 0.0f) {
        return computeMaxAbsDivergence(uField, vField);
    }

    buildRhs(etaField, uField, vField, dt);
    solveCoupledPressure(dt);
    applyPressureGradient(uField, vField, dt);
    applyEtaCorrection(etaField, uField, vField, dt);
    return computeMaxAbsDivergence(uField, vField);
}

const std::vector<float>& PressureSolver::pressure() const {
    return p;
}

int PressureSolver::idxCell(int i, int j) const {
    return i * N + j;
}

int PressureSolver::idxU(int i, int jFace) const {
    return i * (N + 1) + jFace;
}

int PressureSolver::idxV(int iFace, int j) const {
    return iFace * N + j;
}

void PressureSolver::buildRhs(
    const std::vector<float>& etaField,
    const std::vector<float>& uField,
    const std::vector<float>& vField,
    float dt
) {
    const float hydroScale = rho * g;
    const float fluxScale = meanDepth * dt;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const int id = idxCell(i, j);
            const float divU = divergenceAtCell(uField, vField, i, j);
            // Schur-style RHS from linearized eta-q coupling:
            // q - g*dt^2*Lap(q) = rho*g*(eta* - dt*H*div(u*))
            rhs[id] = hydroScale * (etaField[id] - fluxScale * divU);
        }
    }
}

void PressureSolver::solveCoupledPressure(float dt) {
    // Matrix-free PCG for A(q)=rhs where A = I - g*dt^2*L
    // and L = D*G uses matched MAC operators.
    applyCoupledOperator(p, cgAp, dt);
    for (int i = 0; i < N * N; ++i) {
        cgR[i] = rhs[i] - cgAp[i];
        cgP[i] = cgR[i];
    }

    float rrOld = dotCells(cgR, cgR);
    if (rrOld < tolerance * tolerance) {
        return;
    }

    for (int it = 0; it < maxIterations; ++it) {
        applyCoupledOperator(cgP, cgAp, dt);
        const float denom = std::max(dotCells(cgP, cgAp), 1e-20f);
        const float alpha = rrOld / denom;

        for (int i = 0; i < N * N; ++i) {
            p[i] += alpha * cgP[i];
            cgR[i] -= alpha * cgAp[i];
        }

        const float rrNew = dotCells(cgR, cgR);
        if (rrNew < tolerance * tolerance) {
            break;
        }

        const float beta = rrNew / std::max(rrOld, 1e-20f);
        for (int i = 0; i < N * N; ++i) {
            cgP[i] = cgR[i] + beta * cgP[i];
        }
        rrOld = rrNew;
    }
}

void PressureSolver::applyPressureGradient(
    std::vector<float>& uField,
    std::vector<float>& vField,
    float dt
) const {
    const float scale = (dt / (rho * meanDepth)) * nhStrength;

    // u lives on vertical faces: j = 0..N
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            const float dpdx = gradXAtUFace(p, i, j);
            uField[idxU(i, j)] -= scale * dpdx;
        }
        uField[idxU(i, 0)] = 0.0f;
        uField[idxU(i, N)] = 0.0f;
    }

    // v lives on horizontal faces: i = 0..N
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float dpdy = gradYAtVFace(p, i, j);
            vField[idxV(i, j)] -= scale * dpdy;
        }
    }
    for (int j = 0; j < N; ++j) {
        vField[idxV(0, j)] = 0.0f;
        vField[idxV(N, j)] = 0.0f;
    }
}

void PressureSolver::applyEtaCorrection(
    std::vector<float>& etaField,
    const std::vector<float>& uField,
    const std::vector<float>& vField,
    float dt
) const {
    // Continuity-consistent eta update with corrected velocity.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const int id = idxCell(i, j);
            const float divU = divergenceAtCell(uField, vField, i, j);
            etaField[id] -= dt * meanDepth * divU;
        }
    }
}

float PressureSolver::computeMaxAbsDivergence(
    const std::vector<float>& uField,
    const std::vector<float>& vField
) const {
    float maxAbsDiv = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            maxAbsDiv = std::max(maxAbsDiv, std::fabs(divergenceAtCell(uField, vField, i, j)));
        }
    }
    return maxAbsDiv;
}

float PressureSolver::divergenceAtCell(
    const std::vector<float>& uField,
    const std::vector<float>& vField,
    int i,
    int j
) const {
    const float dudx = (uField[idxU(i, j + 1)] - uField[idxU(i, j)]) / dx;
    const float dvdy = (vField[idxV(i + 1, j)] - vField[idxV(i, j)]) / dy;
    return dudx + dvdy;
}

float PressureSolver::gradXAtUFace(const std::vector<float>& cellField, int i, int jFace) const {
    return (cellField[idxCell(i, jFace)] - cellField[idxCell(i, jFace - 1)]) / dx;
}

float PressureSolver::gradYAtVFace(const std::vector<float>& cellField, int iFace, int j) const {
    return (cellField[idxCell(iFace, j)] - cellField[idxCell(iFace - 1, j)]) / dy;
}

float PressureSolver::applyCoupledOperatorAtCell(
    const std::vector<float>& qField,
    float dt,
    int i,
    int j
) const {
    const float qC = qField[idxCell(i, j)];
    const float qW = qField[idxCell(i, (j > 0) ? (j - 1) : j)];
    const float qE = qField[idxCell(i, (j + 1 < N) ? (j + 1) : j)];
    const float qS = qField[idxCell((i > 0) ? (i - 1) : i, j)];
    const float qN = qField[idxCell((i + 1 < N) ? (i + 1) : i, j)];
    const float lap = (qE - 2.0f * qC + qW) / (dx * dx) +
                      (qN - 2.0f * qC + qS) / (dy * dy);
    return qC - g * dt * dt * lap;
}

void PressureSolver::applyCoupledOperator(
    const std::vector<float>& in,
    std::vector<float>& out,
    float dt
) const {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out[idxCell(i, j)] = applyCoupledOperatorAtCell(in, dt, i, j);
        }
    }
}

float PressureSolver::dotCells(const std::vector<float>& a, const std::vector<float>& b) const {
    float acc = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}