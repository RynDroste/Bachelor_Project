#include "render/scene_camera.h"

#include "render/boat.h"

#include <GLFW/glfw3.h>
#include <cmath>

namespace {

constexpr float kRotateSens   = 0.0022f;
constexpr float kFpsMoveSpeed = 28.f;
constexpr float kPitchLimit   = 1.48f;
constexpr float kOrbitRMin    = 8.f;
constexpr float kOrbitRMax    = 280.f;

glm::vec3 boatPivot(const Boat& boat) {
    return glm::vec3(boat.pos.x, boat.z + 1.1f, boat.pos.y);
}

glm::vec3 orbitFocus(const Boat& boat) {
    return boatPivot(boat) + glm::vec3(0.f, 1.6f, 0.f);
}

}  // namespace

void SceneCamera::resetOrbitalFromEye(const glm::vec3& worldEye, const Boat& boat) {
    const glm::vec3 focus = orbitFocus(boat);
    glm::vec3       d     = worldEye - focus;
    float           len   = glm::length(d);
    if (len < 1e-3f) {
        d   = glm::vec3(40.f, 25.f, 40.f);
        len = glm::length(d);
    }
    orbitRadius_ = glm::clamp(len, kOrbitRMin, kOrbitRMax);
    orbitPitch_  = std::asin(glm::clamp(d.y / orbitRadius_, -0.99f, 0.99f));
    orbitYaw_    = std::atan2(d.z, d.x);
    mode         = SceneCamMode::Orbital;
    recomputeOrbital(boat);
}

void SceneCamera::setMode(SceneCamMode m, const Boat& boat) {
    if (m == mode) {
        if (m == SceneCamMode::Orbital)
            recomputeOrbital(boat);
        return;
    }

    const SceneCamMode prev = mode;

    if (m == SceneCamMode::Fps) {
        if (prev == SceneCamMode::Orbital)
            syncFpsFromOrbital(boat);
    } else if (m == SceneCamMode::Orbital) {
        if (prev == SceneCamMode::Fps)
            syncOrbitalFromFps(boat);
    }

    mode        = m;
    dragActive_ = false;

    if (mode == SceneCamMode::Orbital)
        recomputeOrbital(boat);
    else {
        eye_    = fpsEye_;
        target_ = fpsEye_ + glm::vec3(std::cos(fpsYaw_) * std::cos(fpsPitch_), std::sin(fpsPitch_),
                                       std::sin(fpsYaw_) * std::cos(fpsPitch_));
    }
}

void SceneCamera::syncFpsFromOrbital(const Boat& boat) {
    recomputeOrbital(boat);
    fpsEye_ = eye_;
    glm::vec3 f = glm::normalize(target_ - eye_);
    fpsPitch_   = std::asin(glm::clamp(f.y, -1.f, 1.f));
    fpsYaw_     = std::atan2(f.z, f.x);
}

void SceneCamera::syncOrbitalFromFps(const Boat& boat) {
    const glm::vec3 focus = orbitFocus(boat);
    glm::vec3       d     = fpsEye_ - focus;
    float           len   = glm::length(d);
    if (len < 1e-3f) {
        orbitRadius_ = 50.f;
        orbitPitch_  = 0.3f;
        orbitYaw_    = 0.f;
        return;
    }
    orbitRadius_ = glm::clamp(len, kOrbitRMin, kOrbitRMax);
    orbitPitch_  = std::asin(glm::clamp(d.y / len, -0.99f, 0.99f));
    orbitYaw_    = std::atan2(d.z, d.x);
}

void SceneCamera::recomputeOrbital(const Boat& boat) {
    const glm::vec3 focus = orbitFocus(boat);
    orbitPitch_           = glm::clamp(orbitPitch_, -kPitchLimit, kPitchLimit);
    const float         cosp = std::cos(orbitPitch_);
    glm::vec3 offset(cosp * std::cos(orbitYaw_), std::sin(orbitPitch_), cosp * std::sin(orbitYaw_));
    eye_    = focus + orbitRadius_ * offset;
    target_ = focus;
}

void SceneCamera::update(GLFWwindow* w, float dt, const Boat& boat, bool imguiWantMouse, bool imguiWantKb,
                         float imguiMouseWheel) {
    double mx, my;
    glfwGetCursorPos(w, &mx, &my);

    const bool wantGameMouse = !imguiWantMouse;
    const bool wantGameKb    = !imguiWantKb;
    const bool dragNow       = wantGameMouse && glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

    float dx = 0.f, dy = 0.f;
    if (dragNow) {
        if (!dragActive_) {
            lastMx_ = mx;
            lastMy_ = my;
        } else {
            dx = static_cast<float>(mx - lastMx_);
            dy = static_cast<float>(my - lastMy_);
        }
    }
    lastMx_     = mx;
    lastMy_     = my;
    dragActive_ = dragNow;

    if (mode == SceneCamMode::Orbital) {
        if (!imguiWantMouse && std::abs(imguiMouseWheel) > 1e-6f) {
            const float factor = std::exp(-imguiMouseWheel * 0.15f);
            orbitRadius_       = glm::clamp(orbitRadius_ * factor, kOrbitRMin, kOrbitRMax);
        }
        if (dragNow && dx * dx + dy * dy > 0.f) {
            orbitYaw_ += dx * kRotateSens;
            orbitPitch_ += dy * kRotateSens;
        }
        recomputeOrbital(boat);
        return;
    }

    // Fps
    if (dragNow && dx * dx + dy * dy > 0.f) {
        fpsYaw_ -= dx * kRotateSens;
        fpsPitch_ += dy * kRotateSens;
        fpsPitch_ = glm::clamp(fpsPitch_, -kPitchLimit, kPitchLimit);
    }

    glm::vec3 forward(std::cos(fpsYaw_) * std::cos(fpsPitch_), std::sin(fpsPitch_),
                      std::sin(fpsYaw_) * std::cos(fpsPitch_));
    forward = glm::normalize(forward);
    glm::vec3 worldUp(0.f, 1.f, 0.f);
    glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));
    if (glm::length(right) < 1e-4f)
        right = glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 up = glm::normalize(glm::cross(right, forward));

    float move = kFpsMoveSpeed * dt;
    if (wantGameKb && glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        move *= 3.f;
    if (wantGameKb && glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
        move *= 0.25f;

    if (wantGameKb) {
        if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS)
            fpsEye_ += forward * move;
        if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS)
            fpsEye_ -= forward * move;
        if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS)
            fpsEye_ += right * move;
        if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS)
            fpsEye_ -= right * move;
        if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS)
            fpsEye_ += up * move;
        if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS)
            fpsEye_ -= up * move;
    }

    eye_    = fpsEye_;
    target_ = fpsEye_ + forward * 10.f;
}
