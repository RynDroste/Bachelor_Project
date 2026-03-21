# Bachelor_Project

# Fluid Simulation and Real-Time Rendering of Shallow Water Equations

| Status | Week | Planned Activity | Expected Result |
| :--- | :--- | :--- | :--- |
| ✅ | W1 | Setup environment and basic shaders | The screen can render a static plane with simple colors. |
| ✅ | W2 | Implement SWE base update cycle | The fundamental changes in the height field can be observed. |
| ⬜ | W3 | Staggered grid & advection implementation | Stable fluid motion |
| ⬜ | W4 | CFL condition & fluctuation handling | Numerically stable simulation |
| ⬜ | W5 | Blinn-Phong shading implementation | Giving the waveform a sense of depth and light and shadow feedback. |
| ⬜ | W6 | Real-time vertex height displacement | Dynamic water mesh |
| ⬜ | W7 | Implement Boundary conditions | Constrained fluid behavior |
| ⬜ | W8 | Environment mapping & Fresnel effect | Enhanced water aesthetics |
| ⬜ | W9 | Solid-fluid coupling | Interaction with dynamic objects |
| ⬜ | W10 | Particle system / Foam map | Visual detail enrichment |
| ⬜ | W11 | Performance optimization | Real-time frame rate stability |
| ⬜ | W12 | Final documentation | Completed Thesis Report |

Initialization Phase

│
├─ Load terrain b(x,y)

├─ Calculate global depth field h(x,y) = η₀ - b(x,y)

├─ Initialize SWE mesh (b as input)

└─ Pre-compute depth lookup table for Airy layer

──────────────────────────────── Frame-by-Frame Loop

│
├─ [1] SWE step (terrain involved throughout)

│
├─ hydrostatic reconstruction (using b)

│
├─ Dry/wet interface update

│
└─ Output: η_SWE, u_SWE (large-scale flow field including terrain influence)

│
├─ [2] Airy step (terrain as depth correction)

│
├─ Extract current depth from η_SWE h = η_SWE - b

│
├─ FFT → Frequency domain phase evolution (using h) Correcting the dispersion relation)

│ ├─ Injecting local disturbance sources (ship, raindrops)

│ └─ IFFT → η_Airy

│ ├─ [3] Tessendorf update (terrain as a mask)

│ ├─ Evolving according to statistical spectrum (unaware of terrain details)

│ ├─ Amplitude attenuation using depth field: h < threshold → suppression

│ └─ Output: η_Tess

│ └─ [4] Synthesizing the final wavefront

η_final = η_SWE + w_Airy(h)·η_Airy + w_Tess(h)·η_Tess

↑ ↑
Shallow water weight → 0 Shallow water weight → 0