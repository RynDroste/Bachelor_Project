#include "shallow_water_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>

ShallowWaterSolver::ShallowWaterSolver(int gridSize)
    : gridSize_(gridSize),
      N_(gridSize + 1),
      dx_(2.0f / static_cast<float>(gridSize)),
      dy_(dx_),
      targetDt_(1.0f / 240.0f),
      H_(1.0f),
      g_(9.81f),
      f_(0.1f),
      linearDrag_(0.2f),
      energyThreshold_(5e-4f),
      lowEnergyStepsRequired_(180),
      dt_(0.0f),
      dampingFactor_(1.0f),
      etaCurr_(N_ * N_, 0.0f),
      etaNext_(N_ * N_, 0.0f),
      uCurr_(N_ * (N_ + 1), 0.0f),
      uNext_(N_ * (N_ + 1), 0.0f),
      vCurr_((N_ + 1) * N_, 0.0f),
      vNext_((N_ + 1) * N_, 0.0f),
      accumulator_(0.0f),
      lowEnergySteps_(0),
      simulationActive_(true) {
    const float etaAmplitude = 0.2f;
    const float sigmaCells = 6.0f;
    const float twoSigma2 = 2.0f * sigmaCells * sigmaCells;
    const int centerI = N_ / 2;
    const int centerJ = N_ / 2;

    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            const float di = static_cast<float>(i - centerI);
            const float dj = static_cast<float>(j - centerJ);
            const float r2 = di * di + dj * dj;
            etaCurr_[idxEta(i, j)] = etaAmplitude * std::exp(-r2 / twoSigma2);
        }
    }

    const float waveC = std::sqrt(g_ * H_);
    const float invCellNorm = std::sqrt((1.0f / (dx_ * dx_)) + (1.0f / (dy_ * dy_)));
    const float cflLimit = 0.9f;
    const float maxStableDt = cflLimit / (waveC * invCellNorm);
    dt_ = std::min(targetDt_, maxStableDt);
    const float cfl = waveC * dt_ * invCellNorm;
    dampingFactor_ = std::exp(-linearDrag_ * dt_);

    if (dt_ < targetDt_) {
        std::cout << "CFL-limited dt: " << targetDt_ << " -> " << dt_ << " (CFL = " << cfl << ")\n";
    }
}

void ShallowWaterSolver::advance(float frameDt) {
    if (frameDt > 0.1f) {
        frameDt = 0.1f;
    }
    accumulator_ += frameDt;

    while (accumulator_ >= dt_) {
        if (simulationActive_) {
            step();
        }
        accumulator_ -= dt_;
    }
}

float ShallowWaterSolver::etaAt(int i, int j) const {
    return etaCurr_[idxEta(i, j)];
}

int ShallowWaterSolver::resolution() const {
    return N_;
}

int ShallowWaterSolver::idxEta(int i, int j) const {
    return i * N_ + j;
}

int ShallowWaterSolver::idxU(int i, int jFace) const {
    return i * (N_ + 1) + jFace;
}

int ShallowWaterSolver::idxV(int iFace, int j) const {
    return iFace * N_ + j;
}

void ShallowWaterSolver::step() {
    for (int i = 0; i < N_; ++i) {
        for (int j = 1; j < N_; ++j) {
            const float etaDx = (etaCurr_[idxEta(i, j)] - etaCurr_[idxEta(i, j - 1)]) / dx_;
            const float vInterp = 0.25f * (
                vCurr_[idxV(i, j - 1)] + vCurr_[idxV(i, j)] +
                vCurr_[idxV(i + 1, j - 1)] + vCurr_[idxV(i + 1, j)]
            );
            uNext_[idxU(i, j)] = uCurr_[idxU(i, j)] + dt_ * (-g_ * etaDx + f_ * vInterp);
        }
    }

    for (int i = 1; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            const float etaDy = (etaCurr_[idxEta(i, j)] - etaCurr_[idxEta(i - 1, j)]) / dy_;
            const float uInterp = 0.25f * (
                uCurr_[idxU(i - 1, j)] + uCurr_[idxU(i - 1, j + 1)] +
                uCurr_[idxU(i, j)] + uCurr_[idxU(i, j + 1)]
            );
            vNext_[idxV(i, j)] = vCurr_[idxV(i, j)] + dt_ * (-g_ * etaDy - f_ * uInterp);
        }
    }

    for (int i = 0; i < N_; ++i) {
        for (int j = 1; j < N_; ++j) {
            uNext_[idxU(i, j)] *= dampingFactor_;
        }
    }

    for (int i = 1; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            vNext_[idxV(i, j)] *= dampingFactor_;
        }
    }

    for (int i = 0; i < N_; ++i) {
        uNext_[idxU(i, 0)] = 0.0f;
        uNext_[idxU(i, N_)] = 0.0f;
    }

    for (int j = 0; j < N_; ++j) {
        vNext_[idxV(0, j)] = 0.0f;
        vNext_[idxV(N_, j)] = 0.0f;
    }

    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            const float dudx = (uNext_[idxU(i, j + 1)] - uNext_[idxU(i, j)]) / dx_;
            const float dvdy = (vNext_[idxV(i + 1, j)] - vNext_[idxV(i, j)]) / dy_;
            etaNext_[idxEta(i, j)] = etaCurr_[idxEta(i, j)] - dt_ * H_ * (dudx + dvdy);
        }
    }

    etaCurr_.swap(etaNext_);
    uCurr_.swap(uNext_);
    vCurr_.swap(vNext_);

    float kineticEnergy = 0.0f;
    float potentialEnergy = 0.0f;
    for (int i = 0; i < N_; ++i) {
        for (int j = 1; j < N_; ++j) {
            const float uVal = uCurr_[idxU(i, j)];
            kineticEnergy += 0.5f * H_ * uVal * uVal;
        }
    }
    for (int i = 1; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            const float vVal = vCurr_[idxV(i, j)];
            kineticEnergy += 0.5f * H_ * vVal * vVal;
        }
    }
    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < N_; ++j) {
            const float etaVal = etaCurr_[idxEta(i, j)];
            potentialEnergy += 0.5f * g_ * etaVal * etaVal;
        }
    }

    const float totalEnergy = (kineticEnergy + potentialEnergy) * dx_ * dy_;
    if (totalEnergy < energyThreshold_) {
        ++lowEnergySteps_;
        if (lowEnergySteps_ >= lowEnergyStepsRequired_) {
            simulationActive_ = false;
            std::cout << "Simulation stopped: energy below threshold (E = " << totalEnergy << ")\n";
        }
    } else {
        lowEnergySteps_ = 0;
    }
}
