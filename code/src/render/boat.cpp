#include "render/boat.h"

#include "solver_pipeline/shallow_water_solver.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kDryEps   = 1e-4f;
constexpr float kMaxSpeed = 16.f;
constexpr float kTurnGain = 1.2f;
// Momentum injection into SWE (Qx/Qy); lower = weaker wake / less grid ringing.
constexpr float kPointQStrength = 840.f;
// Twin stern jets: each branch ± this yaw from hull axis (deg). Between ~15–30 gives a V wake.
constexpr float kTwinJetYawDeg = 22.5f;

float sampleH(const Grid& g, int i, int j) {
    i = std::clamp(i, 0, g.NX - 1);
    j = std::clamp(j, 0, g.NY - 1);
    return g.H(i, j);
}

float sampleEta(const Grid& g, int i, int j) {
    i = std::clamp(i, 0, g.NX - 1);
    j = std::clamp(j, 0, g.NY - 1);
    return g.B(i, j) + g.H(i, j);
}

float bilinearCell(const Grid& g, float cx, float cy, bool surface) {
    cx = std::clamp(cx, 0.f, static_cast<float>(g.NX - 1) - 1e-4f);
    cy = std::clamp(cy, 0.f, static_cast<float>(g.NY - 1) - 1e-4f);
    int   i0 = static_cast<int>(cx);
    int   j0 = static_cast<int>(cy);
    float fx = cx - static_cast<float>(i0);
    float fy = cy - static_cast<float>(j0);
    i0 = std::min(i0, g.NX - 2);
    j0 = std::min(j0, g.NY - 2);

    auto samp = [&](int ii, int jj) {
        return surface ? sampleEta(g, ii, jj) : sampleH(g, ii, jj);
    };

    float a = samp(i0, j0);
    float b = samp(i0 + 1, j0);
    float c = samp(i0, j0 + 1);
    float d = samp(i0 + 1, j0 + 1);
    float ab = a + fx * (b - a);
    float cd = c + fx * (d - c);
    return ab + fy * (cd - ab);
}

void worldToCellFrac(float x, float z, float halfW, float halfD, float dx, float& cx, float& cy) {
    float u = (x + halfW) / dx - 0.5f;
    float v = (z + halfD) / dx - 0.5f;
    cx = u;
    cy = v;
}

}  // namespace

void updateBoat(Boat& boat, Grid& g, GLFWwindow* window, glm::vec2 sweCenterXZ, float halfW, float halfD,
                float restEta, float dt, bool keyboardSteeringEnabled, bool useArrowKeysForSteering) {
    if (keyboardSteeringEnabled && window) {
        if (useArrowKeysForSteering) {
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
                boat.throttle = std::min(1.f, boat.throttle + dt * 0.8f);
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
                boat.throttle = std::max(-1.f, boat.throttle - dt * 0.8f);
            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
                boat.rudder = std::max(-0.5f, boat.rudder - dt * 0.6f);
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
                boat.rudder = std::min(0.5f, boat.rudder + dt * 0.6f);
            if (glfwGetKey(window, GLFW_KEY_LEFT) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_RIGHT) != GLFW_PRESS)
                boat.rudder *= std::exp(-dt * 4.f);
        } else {
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                boat.throttle = std::min(1.f, boat.throttle + dt * 0.8f);
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                boat.throttle = std::max(-1.f, boat.throttle - dt * 0.8f);
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                boat.rudder = std::max(-0.5f, boat.rudder - dt * 0.6f);
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                boat.rudder = std::min(0.5f, boat.rudder + dt * 0.6f);
            if (glfwGetKey(window, GLFW_KEY_A) != GLFW_PRESS && glfwGetKey(window, GLFW_KEY_D) != GLFW_PRESS)
                boat.rudder *= std::exp(-dt * 4.f);
        }
    } else {
        boat.rudder *= std::exp(-dt * 4.f);
    }

    boat.speed = boat.throttle * kMaxSpeed;
    const float steerSign = (boat.speed >= 0.f) ? 1.f : -1.f;
    float turnRate = boat.rudder * kTurnGain * steerSign *
                     (std::abs(boat.speed) / std::max(boat.length, 0.1f));
    boat.heading += turnRate * dt;

    float c = std::cos(boat.heading);
    float s = std::sin(boat.heading);
    boat.pos.x += c * boat.speed * dt;
    boat.pos.y += s * boat.speed * dt;

    // Sample water height from SWE only if inside the (moving) simulation window.
    const float localX = boat.pos.x - sweCenterXZ.x;
    const float localZ = boat.pos.y - sweCenterXZ.y;
    if (localX > -halfW && localX < halfW && localZ > -halfD && localZ < halfD) {
        float cx, cy;
        worldToCellFrac(localX, localZ, halfW, halfD, g.dx, cx, cy);
        boat.z = bilinearCell(g, cx, cy, true);
    } else {
        boat.z = restEta;
    }
}

void applyBoatForcing(Boat& boat, Grid& g, glm::vec2 sweCenterXZ, float halfW, float halfD, float dt) {
    // Convert to SWE-local coordinates; forcing only happens when the boat is inside the window.
    const float localX = boat.pos.x - sweCenterXZ.x;
    const float localZ = boat.pos.y - sweCenterXZ.y;
    if (localX < -halfW || localX > halfW || localZ < -halfD || localZ > halfD)
        return;

    const float th = boat.heading;
    const float vx = std::cos(th) * boat.speed;
    const float vz = std::sin(th) * boat.speed;
    const float vlen = std::sqrt(vx * vx + vz * vz);
    if (vlen < 1e-4f)
        return;

    const float cx = (localX + halfW) / g.dx - 0.5f;
    const float cy = (localZ + halfD) / g.dx - 0.5f;
    const int ic = std::clamp(static_cast<int>(std::floor(cx)), 1, g.NX - 2);
    const int jc = std::clamp(static_cast<int>(std::floor(cy)), 1, g.NY - 2);

    if (g.H(ic, jc) < kDryEps)
        return;

    const float dq = kPointQStrength * boat.draft * vlen * dt;

    // Hull-axis bisector for jet directions: forward along heading, astern uses opposite axis
    // so reverse travel still models stern wash in hull frame.
    constexpr float kPi = 3.14159265f;
    const float       ax  = (boat.speed >= 0.f) ? th : th + kPi;
    const float       yaw = kTwinJetYawDeg * (kPi / 180.f);
    const float       dirLx = std::cos(ax + yaw);
    const float       dirLz = std::sin(ax + yaw);
    const float       dirRx = std::cos(ax - yaw);
    const float       dirRz = std::sin(ax - yaw);

    // Spread the forcing over an elliptical Gaussian footprint matching the boat's
    // hull (length along heading, width across). A single-cell point source produced
    // a 1-pixel-wide foam streak; using a finite footprint gives the wake real width.
    const float halfLen = std::max(0.5f * boat.length, g.dx);
    const float halfWid = std::max(0.5f * boat.width, g.dx);
    const float cosH = std::cos(th);
    const float sinH = std::sin(th);
    const int   rad  = static_cast<int>(std::ceil(halfLen / g.dx)) + 1;
    const float invHalfLen2 = 1.f / (halfLen * halfLen);
    const float invHalfWid2 = 1.f / (halfWid * halfWid);

    float wsum = 0.f;
    for (int dj = -rad; dj <= rad; ++dj) {
        for (int di = -rad; di <= rad; ++di) {
            const float lx =  cosH * (di * g.dx) + sinH * (dj * g.dx);
            const float ly = -sinH * (di * g.dx) + cosH * (dj * g.dx);
            const float r2 = lx * lx * invHalfLen2 + ly * ly * invHalfWid2;
            if (r2 > 1.f)
                continue;
            wsum += std::exp(-3.f * r2);
        }
    }
    if (wsum < 1e-6f)
        return;
    const float invWsum = 1.f / wsum;

    for (int dj = -rad; dj <= rad; ++dj) {
        for (int di = -rad; di <= rad; ++di) {
            const int ii = ic + di;
            const int jj = jc + dj;
            if (ii < 1 || ii >= g.NX - 1 || jj < 1 || jj >= g.NY - 1)
                continue;
            if (g.H(ii, jj) < kDryEps)
                continue;
            const float lx =  cosH * (di * g.dx) + sinH * (dj * g.dx);
            const float ly = -sinH * (di * g.dx) + cosH * (dj * g.dx);
            const float r2 = lx * lx * invHalfLen2 + ly * ly * invHalfWid2;
            if (r2 > 1.f)
                continue;
            const float w = std::exp(-3.f * r2) * invWsum;
            const float dqHalf = 0.5f * w * dq;

            auto inject = [&](float jx, float jz, float mag) {
                if (jx >= 0.f) {
                    g.QX(ii + 1, jj) += mag * jx;
                } else {
                    g.QX(ii, jj) += mag * jx;
                }
                if (jz >= 0.f) {
                    g.QY(ii, jj + 1) += mag * jz;
                } else {
                    g.QY(ii, jj) += mag * jz;
                }
            };
            inject(dirLx, dirLz, dqHalf);
            inject(dirRx, dirRz, dqHalf);
        }
    }
}
