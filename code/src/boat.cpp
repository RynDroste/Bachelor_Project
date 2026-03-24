#include "boat.h"

#include "shallow_water_solver.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kG            = 9.81f;
constexpr float kDryEps       = 1e-4f;
constexpr float kMaxSpeed     = 8.f;
constexpr float kForcingScale = 8.5f;
constexpr float kTurnGain     = 1.2f;
constexpr float kGaussInvScaleAlong  = 0.95f;
constexpr float kGaussInvScaleAcross = 5.5f;

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

void updateBoat(Boat& boat, Grid& g, GLFWwindow* window, float halfW, float halfD, float dt,
                bool manualControl) {
    if (manualControl && window) {
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
    } else {
        boat.rudder *= std::exp(-dt * 2.f);
        boat.throttle = 0.55f;
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

    {
        const float hL = boat.length * 0.5f;
        const float hW = boat.width * 0.5f;
        const float ext = std::sqrt(hL * hL + hW * hW);
        const float pad = g.dx;
        float xmin = -halfW + ext + pad;
        float xmax = halfW - ext - pad;
        float zmin = -halfD + ext + pad;
        float zmax = halfD - ext - pad;
        if (xmin > xmax)
            xmin = xmax = 0.f;
        if (zmin > zmax)
            zmin = zmax = 0.f;
        boat.pos.x = std::clamp(boat.pos.x, xmin, xmax);
        boat.pos.y = std::clamp(boat.pos.y, zmin, zmax);
    }

    float cx, cy;
    worldToCellFrac(boat.pos.x, boat.pos.y, halfW, halfD, g.dx, cx, cy);
    boat.z = bilinearCell(g, cx, cy, true);
}

void applyBoatForcing(Boat& boat, Grid& g, float halfW, float halfD, float dt) {
    const float halfL = boat.length * 0.5f;
    const float halfWb = boat.width * 0.5f;
    const float bx = boat.pos.x;
    const float bz = boat.pos.y;
    const float th = boat.heading;
    const float co = std::cos(th);
    const float si = std::sin(th);

    auto worldCorner = [&](float lx, float ly) {
        float wx = bx + co * lx - si * ly;
        float wz = bz + si * lx + co * ly;
        return glm::vec2(wx, wz);
    };

    glm::vec2 c00 = worldCorner(-halfL, -halfWb);
    glm::vec2 c01 = worldCorner(-halfL, +halfWb);
    glm::vec2 c10 = worldCorner(+halfL, -halfWb);
    glm::vec2 c11 = worldCorner(+halfL, +halfWb);

    float minX = std::min({c00.x, c01.x, c10.x, c11.x});
    float maxX = std::max({c00.x, c01.x, c10.x, c11.x});
    float minZ = std::min({c00.y, c01.y, c10.y, c11.y});
    float maxZ = std::max({c00.y, c01.y, c10.y, c11.y});

    int j0 = static_cast<int>(std::floor((minZ + halfD) / g.dx - 0.5f));
    int j1 = static_cast<int>(std::ceil((maxZ + halfD) / g.dx - 0.5f));
    int i0 = static_cast<int>(std::floor((minX + halfW) / g.dx - 0.5f));
    int i1 = static_cast<int>(std::ceil((maxX + halfW) / g.dx - 0.5f));

    j0 = std::clamp(j0, 1, g.NY - 2);
    j1 = std::clamp(j1, 1, g.NY - 2);
    i0 = std::clamp(i0, 1, g.NX - 2);
    i1 = std::clamp(i1, 1, g.NX - 2);
    i0 = std::max(1, i0 - 2);
    i1 = std::min(g.NX - 2, i1 + 2);
    j0 = std::max(1, j0 - 2);
    j1 = std::min(g.NY - 2, j1 + 2);

    const float vx = co * boat.speed;
    const float vz = si * boat.speed;
    const float vlen = std::sqrt(vx * vx + vz * vz);
    if (vlen < 1e-4f)
        return;

    const float dirX = vx / vlen;
    const float dirZ = vz / vlen;

    for (int j = j0; j <= j1; ++j) {
        for (int i = i0; i <= i1; ++i) {
            float cellX = (static_cast<float>(i) + 0.5f) * g.dx - halfW;
            float cellZ = (static_cast<float>(j) + 0.5f) * g.dx - halfD;
            float dxw = cellX - bx;
            float dzw = cellZ - bz;
            float lx = co * dxw + si * dzw;
            float ly = -si * dxw + co * dzw;

            const float nx = lx / std::max(halfL, 1e-4f);
            const float ny = ly / std::max(halfWb, 1e-4f);
            const float q =
                kGaussInvScaleAlong * nx * nx + kGaussInvScaleAcross * ny * ny;
            const float shapeW = std::exp(-q);
            if (shapeW < 1e-5f)
                continue;

            float hLoc = g.H(i, j);
            if (hLoc < kDryEps)
                continue;

            const float speedAbs = std::abs(boat.speed);
            float froude = speedAbs / std::sqrt(std::max(kG * hLoc, 1e-6f));
            float forcing = boat.draft * speedAbs * shapeW * kForcingScale;
            forcing *= (0.82f + 0.22f * std::min(froude, 2.5f));

            const float nMotion =
                (lx * std::copysign(1.f, boat.speed)) / std::max(halfL, 1e-4f);
            const float sternBlend = std::clamp(0.5f - 0.5f * nMotion, 0.f, 1.f);
            const float bowBlend = std::clamp(0.5f + 0.5f * nMotion, 0.f, 1.f);

            const float dStern = forcing * sternBlend * dt;
            const float dBow = forcing * bowBlend * dt;

            if (dirX >= 0.f) {
                g.QX(i + 1, j) -= dirX * dStern;
                g.QX(i, j) += dirX * dBow;
            } else {
                g.QX(i, j) -= dirX * dStern;
                g.QX(i + 1, j) += dirX * dBow;
            }

            if (dirZ >= 0.f) {
                g.QY(i, j + 1) -= dirZ * dStern;
                g.QY(i, j) += dirZ * dBow;
            } else {
                g.QY(i, j) -= dirZ * dStern;
                g.QY(i, j + 1) += dirZ * dBow;
            }
        }
    }
}
