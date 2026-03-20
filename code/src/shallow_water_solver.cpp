#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>

ShallowWaterSolver::ShallowWaterSolver(int gridSize)
    : ShallowWaterSolver(
          gridSize,
          2.0f / static_cast<float>(std::max(gridSize, 1)),
          2.0f / static_cast<float>(std::max(gridSize, 1))
      ) {}

ShallowWaterSolver::ShallowWaterSolver(int gridSize, float dxMeters, float dyMeters)
    : N(gridSize + 1),
      dx(std::max(dxMeters, 1e-6f)),
      dy(std::max(dyMeters, 1e-6f)),
      targetDt(1.0f / 120.0f),
      g(9.81f),
      f(0.1f),
      linearDrag(0.05f),
      cflLimit(0.45f),
      shapiroStrength(0.0f),
      spongeWidthCells(10),
      spongeMaxSigma(0.0f),
      energyThreshold(5e-6f),
      lowEnergyStepsRequired(600),
      dt(0.0f),
      dryDepthThreshold(5e-3f),
      etaCurr(N * N, 0.0f),
      etaNext(N * N, 0.0f),
      etaStage(N * N, 0.0f),
      etaRhs(N * N, 0.0f),
      uCurr(N * (N + 1), 0.0f),
      uNext(N * (N + 1), 0.0f),
      uStage(N * (N + 1), 0.0f),
      uRhs(N * (N + 1), 0.0f),
      vCurr((N + 1) * N, 0.0f),
      vNext((N + 1) * N, 0.0f),
      vStage((N + 1) * N, 0.0f),
      vRhs((N + 1) * N, 0.0f),
      bathymetry(N * N, 0.0f),
      accumulator(0.0f),
      simulationTime(0.0f),
      lowEnergySteps(0),
      simulationActive(true) {

    updateTimeStepFromCfl();
}

void ShallowWaterSolver::setBathymetry(const std::vector<float>& bedElevation) {
    if (bedElevation.size() == bathymetry.size()) {
        bathymetry = bedElevation;
    } else {
        std::fill(bathymetry.begin(), bathymetry.end(), 0.0f);
    }

    initializeFreeSurfaceFromBathymetry();

    std::fill(uCurr.begin(), uCurr.end(), 0.0f);
    std::fill(uNext.begin(), uNext.end(), 0.0f);
    std::fill(uStage.begin(), uStage.end(), 0.0f);
    std::fill(vCurr.begin(), vCurr.end(), 0.0f);
    std::fill(vNext.begin(), vNext.end(), 0.0f);
    std::fill(vStage.begin(), vStage.end(), 0.0f);
    simulationTime = 0.0f;
    lowEnergySteps = 0;
    simulationActive = true;
}

void ShallowWaterSolver::advance(float frameDt) {
    if (frameDt > 0.1f) {
        frameDt = 0.1f;
    }
    accumulator += frameDt;

    while (accumulator >= dt) {
        if (simulationActive) {
            step();
            simulationTime += dt;
        }
        accumulator -= dt;
    }
}

float ShallowWaterSolver::etaAt(int i, int j) const {
    return etaCurr[idxEta(i, j)];
}

int ShallowWaterSolver::resolution() const {
    return N;
}

int ShallowWaterSolver::idxEta(int i, int j) const {
    return i * N + j;
}

int ShallowWaterSolver::idxU(int i, int jFace) const {
    return i * (N + 1) + jFace;
}

int ShallowWaterSolver::idxV(int iFace, int j) const {
    return iFace * N + j;
}

float ShallowWaterSolver::localDepth(const std::vector<float>& etaField, int i, int j) const {
    const float h = etaField[idxEta(i, j)] - bathymetry[idxEta(i, j)];
    return std::max(h, dryDepthThreshold);
}

float ShallowWaterSolver::reconstructedDepthAtUFace(
    const std::vector<float>& etaField,
    int i,
    int jFace
) const {
    if (jFace <= 0 || jFace >= N) {
        return 0.0f;
    }
    const float etaL = etaField[idxEta(i, jFace - 1)];
    const float etaR = etaField[idxEta(i, jFace)];
    const float zL = bathymetry[idxEta(i, jFace - 1)];
    const float zR = bathymetry[idxEta(i, jFace)];
    const float zStar = std::max(zL, zR);
    const float hL = std::max(etaL - zStar, 0.0f);
    const float hR = std::max(etaR - zStar, 0.0f);
    return 0.5f * (hL + hR);
}

float ShallowWaterSolver::reconstructedDepthAtVFace(
    const std::vector<float>& etaField,
    int iFace,
    int j
) const {
    if (iFace <= 0 || iFace >= N) {
        return 0.0f;
    }
    const float etaD = etaField[idxEta(iFace - 1, j)];
    const float etaU = etaField[idxEta(iFace, j)];
    const float zD = bathymetry[idxEta(iFace - 1, j)];
    const float zU = bathymetry[idxEta(iFace, j)];
    const float zStar = std::max(zD, zU);
    const float hD = std::max(etaD - zStar, 0.0f);
    const float hU = std::max(etaU - zStar, 0.0f);
    return 0.5f * (hD + hU);
}

float ShallowWaterSolver::reconstructedEtaGradientAtUFace(
    const std::vector<float>& etaField,
    int i,
    int jFace
) const {
    if (jFace <= 0 || jFace >= N) {
        return 0.0f;
    }
    const float etaL = etaField[idxEta(i, jFace - 1)];
    const float etaR = etaField[idxEta(i, jFace)];
    const float zL = bathymetry[idxEta(i, jFace - 1)];
    const float zR = bathymetry[idxEta(i, jFace)];
    const float zStar = std::max(zL, zR);
    const float hL = std::max(etaL - zStar, 0.0f);
    const float hR = std::max(etaR - zStar, 0.0f);
    const float etaRecL = hL + zStar;
    const float etaRecR = hR + zStar;
    return (etaRecR - etaRecL) / dx;
}

float ShallowWaterSolver::reconstructedEtaGradientAtVFace(
    const std::vector<float>& etaField,
    int iFace,
    int j
) const {
    if (iFace <= 0 || iFace >= N) {
        return 0.0f;
    }
    const float etaD = etaField[idxEta(iFace - 1, j)];
    const float etaU = etaField[idxEta(iFace, j)];
    const float zD = bathymetry[idxEta(iFace - 1, j)];
    const float zU = bathymetry[idxEta(iFace, j)];
    const float zStar = std::max(zD, zU);
    const float hD = std::max(etaD - zStar, 0.0f);
    const float hU = std::max(etaU - zStar, 0.0f);
    const float etaRecD = hD + zStar;
    const float etaRecU = hU + zStar;
    return (etaRecU - etaRecD) / dy;
}

void ShallowWaterSolver::clampEtaToBathymetry(std::vector<float>& etaField) const {
    for (int i = 0; i < N * N; ++i) {
        etaField[i] = std::max(etaField[i], bathymetry[i]);
    }
}

void ShallowWaterSolver::initializeFreeSurfaceFromBathymetry() {
    etaCurr = bathymetry;
    etaNext = bathymetry;
    etaStage = bathymetry;

    const float etaAmplitude = 0.04f;
    const float sigmaCells = 6.0f;
    const float twoSigma2 = 2.0f * sigmaCells * sigmaCells;
    const int centerI = N / 2;
    const int centerJ = N / 2;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float di = static_cast<float>(i - centerI);
            const float dj = static_cast<float>(j - centerJ);
            const float r2 = di * di + dj * dj;
            etaCurr[idxEta(i, j)] += etaAmplitude * std::exp(-r2 / twoSigma2);
        }
    }
    etaNext = etaCurr;
    etaStage = etaCurr;
}

void ShallowWaterSolver::step() {
    updateTimeStepFromCfl();

    // SSP-RK3 stage 1
    computeRhs(etaCurr, uCurr, vCurr, etaRhs, uRhs, vRhs);
    for (int i = 0; i < N * N; ++i) {
        etaStage[i] = etaCurr[i] + dt * etaRhs[i];
    }
    clampEtaToBathymetry(etaStage);
    for (int i = 0; i < N * (N + 1); ++i) {
        uStage[i] = uCurr[i] + dt * uRhs[i];
    }
    for (int i = 0; i < (N + 1) * N; ++i) {
        vStage[i] = vCurr[i] + dt * vRhs[i];
    }
    enforceVelocityBoundaries(uStage, vStage);

    // SSP-RK3 stage 2
    computeRhs(etaStage, uStage, vStage, etaRhs, uRhs, vRhs);
    for (int i = 0; i < N * N; ++i) {
        etaStage[i] = 0.75f * etaCurr[i] + 0.25f * (etaStage[i] + dt * etaRhs[i]);
    }
    clampEtaToBathymetry(etaStage);
    for (int i = 0; i < N * (N + 1); ++i) {
        uStage[i] = 0.75f * uCurr[i] + 0.25f * (uStage[i] + dt * uRhs[i]);
    }
    for (int i = 0; i < (N + 1) * N; ++i) {
        vStage[i] = 0.75f * vCurr[i] + 0.25f * (vStage[i] + dt * vRhs[i]);
    }
    enforceVelocityBoundaries(uStage, vStage);

    // SSP-RK3 stage 3
    computeRhs(etaStage, uStage, vStage, etaRhs, uRhs, vRhs);
    for (int i = 0; i < N * N; ++i) {
        etaNext[i] = (1.0f / 3.0f) * etaCurr[i] + (2.0f / 3.0f) * (etaStage[i] + dt * etaRhs[i]);
    }
    clampEtaToBathymetry(etaNext);
    for (int i = 0; i < N * (N + 1); ++i) {
        uNext[i] = (1.0f / 3.0f) * uCurr[i] + (2.0f / 3.0f) * (uStage[i] + dt * uRhs[i]);
    }
    for (int i = 0; i < (N + 1) * N; ++i) {
        vNext[i] = (1.0f / 3.0f) * vCurr[i] + (2.0f / 3.0f) * (vStage[i] + dt * vRhs[i]);
    }
    enforceVelocityBoundaries(uNext, vNext);
    etaCurr.swap(etaNext);
    uCurr.swap(uNext);
    vCurr.swap(vNext);
    applyShapiroFilter(etaCurr);
    applyBoundarySponge(etaCurr, uCurr, vCurr, dt);
    clampEtaToBathymetry(etaCurr);

    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            const float uVal = uCurr[idxU(i, j)];
            const float hFace = 0.5f * (localDepth(etaCurr, i, j - 1) + localDepth(etaCurr, i, j));
            kineticEnergy += 0.5f * hFace * uVal * uVal;
        }
    }
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float vVal = vCurr[idxV(i, j)];
            const float hFace = 0.5f * (localDepth(etaCurr, i - 1, j) + localDepth(etaCurr, i, j));
            kineticEnergy += 0.5f * hFace * vVal * vVal;
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float hVal = localDepth(etaCurr, i, j);
            potentialEnergy += 0.5f * g * hVal * hVal;
        }
    }

    const float totalEnergy = (kineticEnergy + potentialEnergy) * dx * dy;
    if (totalEnergy < energyThreshold) {
        ++lowEnergySteps;
        if (lowEnergySteps >= lowEnergyStepsRequired) {
            simulationActive = false;
            std::cout << "Simulation stopped: energy below threshold (E = " << totalEnergy << ")\n";
        }
    } else {
        lowEnergySteps = 0;
    }
}

void ShallowWaterSolver::updateTimeStepFromCfl() {
    float maxDepth = dryDepthThreshold;
    float maxAbsU = 0.0f;
    float maxAbsV = 0.0f;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            maxDepth = std::max(maxDepth, localDepth(etaCurr, i, j));
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            maxAbsU = std::max(maxAbsU, std::fabs(uCurr[idxU(i, j)]));
        }
    }
    for (int i = 1; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            maxAbsV = std::max(maxAbsV, std::fabs(vCurr[idxV(i, j)]));
        }
    }

    const float waveC = std::sqrt(g * std::max(maxDepth, dryDepthThreshold));
    const float sx = (waveC + maxAbsU) / dx;
    const float sy = (waveC + maxAbsV) / dy;
    const float denom = std::max(sx + sy, 1e-6f);
    const float maxStableDt = cflLimit / denom;
    const float minDt = targetDt * 0.2f;
    dt = std::clamp(maxStableDt, minDt, targetDt);
}

void ShallowWaterSolver::applyShapiroFilter(std::vector<float>& etaField) const {
    if (shapiroStrength <= 0.0f) {
        return;
    }

    std::vector<float> filtered = etaField;
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            const int c = idxEta(i, j);
            const float lap =
                etaField[idxEta(i + 1, j)] +
                etaField[idxEta(i - 1, j)] +
                etaField[idxEta(i, j + 1)] +
                etaField[idxEta(i, j - 1)] -
                4.0f * etaField[c];
            filtered[c] = etaField[c] + shapiroStrength * lap;
        }
    }
    etaField.swap(filtered);
}

float ShallowWaterSolver::spongeSigma(float distanceToBoundaryCells) const {
    if (spongeWidthCells <= 0 || spongeMaxSigma <= 0.0f) {
        return 0.0f;
    }
    if (distanceToBoundaryCells >= static_cast<float>(spongeWidthCells)) {
        return 0.0f;
    }
    const float ramp =
        (static_cast<float>(spongeWidthCells) - distanceToBoundaryCells) /
        static_cast<float>(spongeWidthCells);
    return spongeMaxSigma * ramp * ramp;
}

void ShallowWaterSolver::applyBoundarySponge(
    std::vector<float>& etaField,
    std::vector<float>& uField,
    std::vector<float>& vField,
    float dtStep
) const {
    if (dtStep <= 0.0f || spongeWidthCells <= 0 || spongeMaxSigma <= 0.0f) {
        return;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const float d = static_cast<float>(
                std::min(std::min(i, N - 1 - i), std::min(j, N - 1 - j))
            );
            const float sigma = spongeSigma(d);
            if (sigma > 0.0f) {
                const float damping = std::exp(-sigma * dtStep);
                etaField[idxEta(i, j)] *= damping;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int jFace = 0; jFace <= N; ++jFace) {
            const float d = static_cast<float>(
                std::min(std::min(i, N - 1 - i), std::min(jFace, N - jFace))
            );
            const float sigma = spongeSigma(d);
            if (sigma > 0.0f) {
                const float damping = std::exp(-sigma * dtStep);
                uField[idxU(i, jFace)] *= damping;
            }
        }
    }

    for (int iFace = 0; iFace <= N; ++iFace) {
        for (int j = 0; j < N; ++j) {
            const float d = static_cast<float>(
                std::min(std::min(iFace, N - iFace), std::min(j, N - 1 - j))
            );
            const float sigma = spongeSigma(d);
            if (sigma > 0.0f) {
                const float damping = std::exp(-sigma * dtStep);
                vField[idxV(iFace, j)] *= damping;
            }
        }
    }

    enforceVelocityBoundaries(uField, vField);
}

void ShallowWaterSolver::computeRhs(
    const std::vector<float>& etaField,
    const std::vector<float>& uField,
    const std::vector<float>& vField,
    std::vector<float>& etaRhsOut,
    std::vector<float>& uRhsOut,
    std::vector<float>& vRhsOut
) const {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const int idEta = idxEta(i, j);
            const float xEta = static_cast<float>(j);
            const float yEta = static_cast<float>(i);

            const float uAtEta = sampleU(uField, xEta, yEta);
            const float vAtEta = sampleV(vField, xEta, yEta);

            const float backX = std::clamp(xEta - (dt / dx) * uAtEta, 0.0f, static_cast<float>(N - 1));
            const float backY = std::clamp(yEta - (dt / dy) * vAtEta, 0.0f, static_cast<float>(N - 1));
            const float etaDepart = sampleEta(etaField, backX, backY);
            const float advEta = (etaDepart - etaField[idEta]) / dt;

            const float hW = reconstructedDepthAtUFace(etaField, i, j);
            const float hE = reconstructedDepthAtUFace(etaField, i, j + 1);
            const float hS = reconstructedDepthAtVFace(etaField, i, j);
            const float hN = reconstructedDepthAtVFace(etaField, i + 1, j);
            const float fluxDiv =
                (hE * uField[idxU(i, j + 1)] - hW * uField[idxU(i, j)]) / dx +
                (hN * vField[idxV(i + 1, j)] - hS * vField[idxV(i, j)]) / dy;
            etaRhsOut[idEta] = advEta - fluxDiv;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j <= N; ++j) {
            const int idU = idxU(i, j);
            if (j == 0 || j == N) {
                uRhsOut[idU] = 0.0f;
                continue;
            }

            const float hFace = reconstructedDepthAtUFace(etaField, i, j);
            const float hLeft = localDepth(etaField, i, j - 1);
            const float hRight = localDepth(etaField, i, j);
            // Allow wetting front propagation: only block a face if both sides are essentially dry.
            if (hFace <= dryDepthThreshold && std::max(hLeft, hRight) <= dryDepthThreshold) {
                uRhsOut[idU] = -uField[idU] / std::max(dt, 1e-6f);
                continue;
            }

            const float xU = static_cast<float>(j) - 0.5f;
            const float yU = static_cast<float>(i);
            const float uAtU = sampleU(uField, xU, yU);
            const float vAtU = sampleV(vField, xU, yU);

            const float backX = std::clamp(xU - (dt / dx) * uAtU, -0.5f, static_cast<float>(N) - 0.5f);
            const float backY = std::clamp(yU - (dt / dy) * vAtU, 0.0f, static_cast<float>(N - 1));
            const float uDepart = sampleU(uField, backX, backY);
            const float advU = (uDepart - uField[idU]) / dt;

            const float etaDxRec = reconstructedEtaGradientAtUFace(etaField, i, j);
            const float source = -g * etaDxRec + f * vAtU - linearDrag * uField[idU];
            uRhsOut[idU] = advU + source;
        }
    }

    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j < N; ++j) {
            const int idV = idxV(i, j);
            if (i == 0 || i == N) {
                vRhsOut[idV] = 0.0f;
                continue;
            }

            const float hFace = reconstructedDepthAtVFace(etaField, i, j);
            const float hDown = localDepth(etaField, i - 1, j);
            const float hUp = localDepth(etaField, i, j);
            // Allow wetting front propagation: only block a face if both sides are essentially dry.
            if (hFace <= dryDepthThreshold && std::max(hDown, hUp) <= dryDepthThreshold) {
                vRhsOut[idV] = -vField[idV] / std::max(dt, 1e-6f);
                continue;
            }

            const float xV = static_cast<float>(j);
            const float yV = static_cast<float>(i) - 0.5f;
            const float uAtV = sampleU(uField, xV, yV);
            const float vAtV = sampleV(vField, xV, yV);

            const float backX = std::clamp(xV - (dt / dx) * uAtV, 0.0f, static_cast<float>(N - 1));
            const float backY = std::clamp(yV - (dt / dy) * vAtV, -0.5f, static_cast<float>(N) - 0.5f);
            const float vDepart = sampleV(vField, backX, backY);
            const float advV = (vDepart - vField[idV]) / dt;

            const float etaDyRec = reconstructedEtaGradientAtVFace(etaField, i, j);
            const float source = -g * etaDyRec - f * uAtV - linearDrag * vField[idV];
            vRhsOut[idV] = advV + source;
        }
    }
}

void ShallowWaterSolver::enforceVelocityBoundaries(std::vector<float>& uField, std::vector<float>& vField) const {
    for (int i = 0; i < N; ++i) {
        uField[idxU(i, 0)] = 0.0f;
        uField[idxU(i, N)] = 0.0f;
    }
    for (int j = 0; j < N; ++j) {
        vField[idxV(0, j)] = 0.0f;
        vField[idxV(N, j)] = 0.0f;
    }
}

float ShallowWaterSolver::sampleEta(const std::vector<float>& etaField, float xEta, float yEta) const {
    return bilinearSample(etaField, N, N, yEta, xEta);
}

float ShallowWaterSolver::sampleU(const std::vector<float>& uField, float xEta, float yEta) const {
    const float rowCoord = yEta;
    const float colCoord = xEta + 0.5f;
    return bilinearSample(uField, N, N + 1, rowCoord, colCoord);
}

float ShallowWaterSolver::sampleV(const std::vector<float>& vField, float xEta, float yEta) const {
    const float rowCoord = yEta + 0.5f;
    const float colCoord = xEta;
    return bilinearSample(vField, N + 1, N, rowCoord, colCoord);
}

float ShallowWaterSolver::bilinearSample(
    const std::vector<float>& field,
    int rows,
    int cols,
    float rowCoord,
    float colCoord
) const {
    const float r = std::clamp(rowCoord, 0.0f, static_cast<float>(rows - 1));
    const float c = std::clamp(colCoord, 0.0f, static_cast<float>(cols - 1));

    const int r0 = static_cast<int>(std::floor(r));
    const int c0 = static_cast<int>(std::floor(c));
    const int r1 = std::min(r0 + 1, rows - 1);
    const int c1 = std::min(c0 + 1, cols - 1);
    const float tr = r - static_cast<float>(r0);
    const float tc = c - static_cast<float>(c0);

    const float f00 = field[r0 * cols + c0];
    const float f01 = field[r0 * cols + c1];
    const float f10 = field[r1 * cols + c0];
    const float f11 = field[r1 * cols + c1];

    const float top = (1.0f - tc) * f00 + tc * f01;
    const float bottom = (1.0f - tc) * f10 + tc * f11;
    return (1.0f - tr) * top + tr * bottom;
}
