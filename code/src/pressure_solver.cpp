#include "pressure_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>

PressureSolver::PressureSolver(int resolution, float dx, float dy)
    : N(resolution),
      dx(dx),
      dy(dy),
      rho(1.0f),
      maxIterations(200),
      tolerance(1e-5f),
      rhs(N * N, 0.0f),
      p(N * N, 0.0f),
      pNext(N * N, 0.0f) {}