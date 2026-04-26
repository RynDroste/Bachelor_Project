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
constexpr float kPointQStrength = 640.f;

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

    const float dirX = vx / vlen;
    const float dirZ = vz / vlen;
    const float dq = kPointQStrength * boat.draft * vlen * dt;

    const int i = ic;
    const int j = jc;

    // QX(i, j) is the right face of cell (i, j); the right wall (cell NX-1)
    // and the implicit left wall (cell 0's left face) are closed and must not
    // receive injected momentum.
    if (dirX >= 0.f) {
        if (i < g.NX - 1)
            g.QX(i, j) += dq * dirX;
    } else {
        if (i > 0)
            g.QX(i - 1, j) += dq * dirX;
    }
    if (dirZ >= 0.f) {
        if (j < g.NY - 1)
            g.QY(i, j) += dq * dirZ;
    } else {
        if (j > 0)
            g.QY(i, j - 1) += dq * dirZ;
    }
}
