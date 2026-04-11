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

void updateBoat(Boat& boat, Grid& g, GLFWwindow* window, float halfW, float halfD, float dt,
                bool keyboardSteeringEnabled, bool useArrowKeysForSteering);

void applyBoatForcing(Boat& boat, Grid& g, float halfW, float halfD, float dt);
