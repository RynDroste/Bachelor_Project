#include "pressure_solver.h"

#include <algorithm>
#include <cmath>

PressureSolver::PressureSolver(int resolution, float dx, float dy)
    : N(resolution),
      dx(dx),
      dy(dy),
      rho(1.0f),
      meanDepth(1.0f),
      maxIterations(60),
      tolerance(1e-5f),
      rhs(N * N, 0.0f),
      p(N * N, 0.0f),
      pNext(N * N, 0.0f) {}

void PressureSolver::setDensity(float rhoValue) {
    rho = std::max(rhoValue, 1e-6f);
}

void PressureSolver::setMeanDepth(float depthValue) {
    meanDepth = std::max(depthValue, 1e-4f);
}

void PressureSolver::setIterations(int maxIters) {
    maxIterations = std::max(maxIters, 1);
}

void PressureSolver::setTolerance(float eps) {
    tolerance = std::max(eps, 1e-9f);
}

float PressureSolver::project(std::vector<float>& uField, std::vector<float>& vField, float dt) {
    if (N <= 1 || dt <= 0.0f) {
        return computeMaxAbsDivergence(uField, vField);
    }

    buildRhs(uField, vField, dt);
    solvePoisson();
    applyPressureGradient(uField, vField, dt);
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
    const std::vector<float>& uField,
    const std::vector<float>& vField,
    float dt
) {
    const float beta = (rho * meanDepth * meanDepth) / (3.0f * dt);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float dudx = (uField[idxU(i, j + 1)] - uField[idxU(i, j)]) / dx;
            const float dvdy = (vField[idxV(i + 1, j)] - vField[idxV(i, j)]) / dy;
            rhs[idxCell(i, j)] = beta * (dudx + dvdy);
        }
    }
}

void PressureSolver::solvePoisson() {
    // Keep previous p as warm start for faster convergence.
    std::fill(pNext.begin(), pNext.end(), 0.0f);

    const float dx2 = dx * dx;
    const float dy2 = dy * dy;
    const float alpha = (meanDepth * meanDepth) / 3.0f;
    const float invDx2 = 1.0f / dx2;
    const float invDy2 = 1.0f / dy2;
    const float centerCoeff = 1.0f + 2.0f * alpha * (invDx2 + invDy2);

    for (int it = 0; it < maxIterations; ++it) {
        float maxChange = 0.0f;

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const int id = idxCell(i, j);

                // Homogeneous Neumann BC by reflecting pressure at boundaries.
                const float pW = p[idxCell(i, (j > 0) ? (j - 1) : j)];
                const float pE = p[idxCell(i, (j + 1 < N) ? (j + 1) : j)];
                const float pS = p[idxCell((i > 0) ? (i - 1) : i, j)];
                const float pN = p[idxCell((i + 1 < N) ? (i + 1) : i, j)];

                const float numer =
                    rhs[id] +
                    alpha * ((pW + pE) * invDx2 + (pS + pN) * invDy2);
                pNext[id] = numer / centerCoeff;

                maxChange = std::max(maxChange, std::fabs(pNext[id] - p[id]));
            }
        }

        p.swap(pNext);
        if (maxChange < tolerance) {
            break;
        }
    }
}

void PressureSolver::applyPressureGradient(
    std::vector<float>& uField,
    std::vector<float>& vField,
    float dt
) const {
    const float scale = dt / (rho * meanDepth);

    // u lives on vertical faces: j = 0..N
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            const float dpdx = (p[idxCell(i, j)] - p[idxCell(i, j - 1)]) / dx;
            uField[idxU(i, j)] -= scale * dpdx;
        }
        uField[idxU(i, 0)] = 0.0f;
        uField[idxU(i, N)] = 0.0f;
    }

    // v lives on horizontal faces: i = 0..N
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float dpdy = (p[idxCell(i, j)] - p[idxCell(i - 1, j)]) / dy;
            vField[idxV(i, j)] -= scale * dpdy;
        }
    }
    for (int j = 0; j < N; ++j) {
        vField[idxV(0, j)] = 0.0f;
        vField[idxV(N, j)] = 0.0f;
    }
}

float PressureSolver::computeMaxAbsDivergence(
    const std::vector<float>& uField,
    const std::vector<float>& vField
) const {
    float maxAbsDiv = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float dudx = (uField[idxU(i, j + 1)] - uField[idxU(i, j)]) / dx;
            const float dvdy = (vField[idxV(i + 1, j)] - vField[idxV(i, j)]) / dy;
            maxAbsDiv = std::max(maxAbsDiv, std::fabs(dudx + dvdy));
        }
    }
    return maxAbsDiv;
}