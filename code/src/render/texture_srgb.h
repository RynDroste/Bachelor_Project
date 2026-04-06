#pragma once

// 2D sRGB8 texture + mips, REPEAT wrap. Returns 0 if load fails.
unsigned int loadSrgbTexture2DFromFile(const char* path);

// 1×1 white (for valid sampler when file missing; keep blend at 0).
unsigned int makeWhiteTexture2D();
