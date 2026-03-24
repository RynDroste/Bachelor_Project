# Bachelor_Project

# Fluid Simulation and Real-Time Rendering of Shallow Water Equations

## Initial Compilation (Windows / PowerShell)

Execute in the repository root directory (modify the path in the example below according to your clone location). Please replace `CMAKE_TOOLCHAIN_FILE` with the path to your local vcpkg; if you don't need vcpkg, you can remove this line.

```powershell
cd C:\Users\AW\Desktop\Bachelor_Project
Remove-Item -Recurse -Force .\build

cmake -S .\code -B .\build -G "Visual Studio 18 2026" -A x64 `

-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build .\build --config Release --parallel

```

The artifacts are usually located in `build\Release\` (e.g., `BachelorProject.exe`, `DamSWE.exe`). CUDA/cuFFT must be installed and compatible with the generator.

```` | Status | Week | Planned Activity | Expected Result |
| :--- | :--- | :--- | :--- |
| ✅ | W1 | Setup environment and basic shaders | The screen can render a static plane with simple colors. |
| ✅ | W2 | Implement SWE base update cycle | The fundamental changes in the height field can be observed. |
| ✅ | W3 | Staggered grid & advection implementation | Stable fluid motion |
| ✅ | W4 | CFL condition & fluctuation handling | Numerically stable simulation |
| ⬜ | W5 | Blinn-Phong shading implementation | Giving the waveform a sense of depth and light and shadow feedback. |
| ⬜ | W6 | Real-time vertex height displacement | Dynamic water mesh |
| ⬜ | W7 | Implement Boundary conditions | Constrained fluid behavior |
| ⬜ | W8 | Environment mapping & Fresnel effect | Enhanced water aesthetics |
| ⬜ | W9 | Solid-fluid coupling | Interaction with dynamic objects |
| ⬜ | W10 | Particle system / Foam map | Visual detail enrichment |
| ⬜ | W11 | Performance optimization | Real-time frame rate stability |
| ⬜ | W12 | Final documentation | Completed Thesis Report |