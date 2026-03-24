# Bachelor_Project

# Fluid Simulation and Real-Time Rendering of Shallow Water Equations

## 首次编译（Windows / PowerShell）

在仓库根目录执行（下例路径请按你的克隆位置修改）。请将 `CMAKE_TOOLCHAIN_FILE` 改为你本机 vcpkg 的路径；若不需要 vcpkg，可去掉该行。

```powershell
cd C:\Users\AW\Desktop\Bachelor_Project
Remove-Item -Recurse -Force .\build

cmake -S .\code -B .\build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake

cmake --build .\build --config Release --parallel
```

产物一般在 `build\Release\`（例如 `BachelorProject.exe`、`DamSWE.exe`）。CUDA / cuFFT 需已安装并与生成器匹配。

| Status | Week | Planned Activity | Expected Result |
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
