# Bachelor Project — Shallow Water Simulation and Real-Time Rendering

## Author: Xuanlin Chen

Real-time simulation and visualization based on the **shallow water equations**.  
This repository is a **C++ / CMake** project: the numerical solver runs in parallel on **CUDA**, and rendering uses **OpenGL**.

---

## Features

- **Height-field, flux-based SWE**  
  - Explicit time integration for water depth and momentum  
  - Flux form on a staggered grid to support mass conservation and boundary treatment  

- **Coupled pipeline**  
  - Stages such as wave decomposition, shallow-water step, **Airy linear waves**, advection, and recombination can be composed in `pipeline`; see `kCoupledStep` in `main.cpp`  

- **Rendering**  
  - Height-field textures, mesh displacement, and simple shading. 

---

## Build

### 1. Configure and build

```powershell
cd C:\Users\AW\Desktop\Bachelor_Project
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue

cmake -S .\code -B .\build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build .\build --config Release --parallel
```

---

## Equations and discretization

### Governing equations

$$
\frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{v}) = 0,\qquad
\frac{\partial \mathbf{v}}{\partial t} = -g \,\nabla \eta + \mathbf{a}_{\mathrm{ext}}
$$

### Typical main-loop stages

1. **Velocity advection** — MacCormack, semi-Lagrangian, etc.  
2. **Height flux update** — finite volume, conservative discretization  
3. **Gravity term** — $-g\nabla\eta$ correction to velocity  
4. **Boundary conditions** — solid-wall reflection, normal velocity constraints, etc.  
5. **Forcing / sources** — injection, volume sources  
6. **Optional post-processing** — e.g. smoothing  

### Height flux update

$$h_{i,j}^{n+1} = h_{i,j}^{n} - \frac{\Delta t}{\Delta x}\bigl(F_{i+1/2,j} - F_{i-1/2,j}\bigr) - \frac{\Delta t}{\Delta y}\bigl(G_{i,j+1/2} - G_{i,j-1/2}\bigr),\quad F = hu,\; G = hv$$

Upwind flux:

$$
F_{i+1/2,j} = \begin{cases} h_{i,j}\,u_{i+1/2,j} & u>0 \\ h_{i+1,j}\,u_{i+1/2,j} & \text{otherwise} \end{cases}
$$

---

## Project roadmap

| Status | Milestone | Goal |
| :--- | :--- | :--- |
| ✅ | **Build & GL baseline** | Ship a runnable app: build system, window, OpenGL context, shader loading, frame loop, and camera. |
| ✅ | **GPU shallow-water core** | Run the shallow-water equations on the GPU with mass-conserving fluxes and staggered velocities. |
| ✅ | **Advection & stability** | Advect the flow and enforce time-step / wave-speed limits so the simulation stays usable. |
| ✅ | **Coupled wave pipeline** | Combine the SWE step with a dispersive or spectral correction and transport so scales interact coherently. |
| ✅ | **Height-field rendering** | Visualize the free surface in real time from simulated depth and bed height. |
| 🔄 | **Water — lighting** | Define sun/key light and how it hits the surface so all water terms share one coherent setup. |
| 🔄 | **Water — diffuse** | Lambertian or rough diffuse response from the surface normal so the body of water reads in shade and light. |
| ⬜ | **Water — specular** | Glossy / specular lobes and Fresnel-aware highlights for sun glints and viewing angle. |
| ⬜ | **Water — normal mapping** | Break up large facets with tiled or procedural normal detail on top of the displaced mesh. |
| ⬜ | **Water — refraction** | Show what is under the surface with chromatic or single-layer refraction of the scene or bed. |
| ⬜ | **Water — transparency** | Control opacity by depth, wet fraction, and Fresnel so shallow vs deep reads correctly. |
| ⬜ | **Optional: Water — caustics** | Light focusing patterns underwater or on the bed from a refractive surface. |
| ⬜ | **Optional: Water — foam** | Shoreline and breaking cues: whitecaps, foam masks driven by speed, curl, or depth. |
| ⬜ | **Optional: Water — flow appearance** | Visualize motion: flow maps, procedural shimmer, or velocity-tinted cues tied to the simulation. |
| ⬜ | **Water — environment reflection** | Skybox or scene reflections on the water with roughness and Fresnel falloff. |
| ⬜ | **Thesis / report** | Finish and submit the written thesis or project report. |

---


