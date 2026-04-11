#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;
struct Boat;

enum class SceneCamMode { Orbital, Bridge, Fps };

// SimShip-style camera: orbit around boat, on-deck view, or free fly. Call update() after ImGui::NewFrame()
// so WantCapture* reflects the current frame state for game input.
struct SceneCamera {
    SceneCamMode mode = SceneCamMode::Orbital;

    void resetOrbitalFromEye(const glm::vec3& worldEye, const Boat& boat);

    void setMode(SceneCamMode m, const Boat& boat);

    void update(GLFWwindow* w, float dt, const Boat& boat, bool imguiWantMouse, bool imguiWantKb,
                float imguiMouseWheel);

    glm::vec3 eye() const { return eye_; }
    glm::vec3 target() const { return target_; }

private:
    void recomputeOrbital(const Boat& boat);
    void recomputeBridge(const Boat& boat);
    void syncFpsFromOrbital(const Boat& boat);
    void syncOrbitalFromFps(const Boat& boat);

    glm::vec3 eye_{0.f};
    glm::vec3 target_{0.f};

    float orbitYaw_   = 0.f;
    float orbitPitch_ = 0.35f;
    float orbitRadius_ = 90.f;

    float bridgeYawOff_   = 0.f;
    float bridgePitchOff_ = 0.f;

    glm::vec3 fpsEye_{0.f};
    float     fpsYaw_   = 0.f;
    float     fpsPitch_ = 0.f;

    double lastMx_ = 0.0;
    double lastMy_ = 0.0;
    bool   dragActive_ = false;
};
