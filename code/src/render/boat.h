#pragma once

#include <glm/glm.hpp>

struct GLFWwindow;
struct Grid;

struct Boat {
    glm::vec2 pos{0.f, 0.f};
    float     heading = 0.f;
    float     speed = 0.f;
    float     z = 0.f;

    float length = 10.f;
    float width = 3.5f;
    float draft = 0.6f;
    float displacement = 12.f;

    float throttle = 0.f;
    float rudder = 0.f;
};

// The boat lives in world space, but the SWE domain is a moving window centred at `sweCenterXZ`
// with half-extents (halfW, halfD). The boat height (`boat.z`) is sampled from the SWE surface
// when inside the window, and falls back to `restEta` otherwise. The boat is free to travel
// beyond the SWE window (no clamp); only its interaction with the SWE grid is gated by the
// window.
void updateBoat(Boat& boat, Grid& g, GLFWwindow* window, glm::vec2 sweCenterXZ, float halfW, float halfD,
                float restEta, float dt, bool keyboardSteeringEnabled, bool useArrowKeysForSteering);

void applyBoatForcing(Boat& boat, Grid& g, glm::vec2 sweCenterXZ, float halfW, float halfD, float dt);
