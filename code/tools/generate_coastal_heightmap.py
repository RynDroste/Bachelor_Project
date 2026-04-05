#!/usr/bin/env python3
"""
Coastal terrain: strictly increasing bed profile along +x (piecewise linear 0..3.9, 4.1..6, 6..8)
plus small Perlin fBm detail; each row is forced monotone in column index via cumulative max.

Outputs
-------
1) .raw  little-endian float32, NX*NY values, C row-major matching C++ Grid::terrain:
      index = i + j * NX   (i = column/x, j = row/y)
      rows j=0..NY-1, each row i=0..NX-1.
2) Optional .png preview (needs pillow): grayscale normalized to 0..255.

Requires numpy; for --png also: pip install pillow

Examples:
  python generate_coastal_heightmap.py --seed 42 -o ../assets/terrain_coastal_128.raw
  python generate_coastal_heightmap.py --nx 256 --ny 256 --png preview.png
"""

from __future__ import annotations

import argparse

import numpy as np

GRID_W_DEFAULT = 256
GRID_H_DEFAULT = 256

SEA_END = 0.60
BEACH_END = 0.90

# Piecewise linear column base z(x): sea [0, SEA_END)->[0,3.9], beach [SEA_END,BEACH_END)->[4.1,6], land ->[6,8]
SEA_Z0 = 0.0
SEA_Z1 = 3.9
BEACH_Z0 = 4.1
BEACH_Z1 = 6.0
LAND_Z0 = 6.0
LAND_Z1 = 8.0

SCALE_DEFAULT = 4.0
OCTAVES_DEFAULT = 6
LACUNARITY_DEFAULT = 2.0
PERSISTENCE_DEFAULT = 0.5
# fBm is ~[-1,1]; scaled before enforcing monotone along +x
NOISE_AMP_DEFAULT = 0.12


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t


def _grad(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D Perlin gradient: pick one of four directions from hash, dot with (x,y)."""
    h = h & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


class Perlin2D:
    def __init__(self, seed: int) -> None:
        p = np.arange(256, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(p)
        self.p = np.concatenate([p, p])

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        xi = (np.floor(x).astype(np.int64) & 255).ravel()
        yi = (np.floor(y).astype(np.int64) & 255).ravel()
        xf = (x - np.floor(x)).ravel()
        yf = (y - np.floor(y)).ravel()
        u = _fade(xf)
        v = _fade(yf)

        p = self.p
        aa = p[p[xi] + yi]
        ab = p[p[xi] + yi + 1]
        ba = p[p[xi + 1] + yi]
        bb = p[p[xi + 1] + yi + 1]

        x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1.0, yf), u)
        x2 = _lerp(_grad(ab, xf, yf - 1.0), _grad(bb, xf - 1.0, yf - 1.0), u)
        out = _lerp(x1, x2, v)
        return out.reshape(x.shape)


def sample_fbm(
    perlin: Perlin2D,
    nx: np.ndarray,
    ny: np.ndarray,
    octaves: int,
    lacunarity: float,
    persistence: float,
) -> np.ndarray:
    value = np.zeros_like(nx, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    for _ in range(octaves):
        value += perlin.noise(nx * frequency, ny * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value / max_value


def column_base_vec(x_ratio: np.ndarray) -> np.ndarray:
    """Strictly increasing piecewise-linear bed elevation vs normalized column x in [0,1)."""
    out = np.zeros_like(x_ratio, dtype=np.float64)
    sea_m = x_ratio < SEA_END
    beach_m = (x_ratio >= SEA_END) & (x_ratio < BEACH_END)
    land_m = x_ratio >= BEACH_END
    out[sea_m] = SEA_Z0 + (SEA_Z1 - SEA_Z0) * (x_ratio[sea_m] / SEA_END)
    tb = (x_ratio[beach_m] - SEA_END) / (BEACH_END - SEA_END)
    out[beach_m] = BEACH_Z0 + (BEACH_Z1 - BEACH_Z0) * tb
    tl = (x_ratio[land_m] - BEACH_END) / (1.0 - BEACH_END)
    out[land_m] = LAND_Z0 + (LAND_Z1 - LAND_Z0) * tl
    return out


def generate_terrain(
    seed: int,
    grid_w: int,
    grid_h: int,
    scale: float = SCALE_DEFAULT,
    octaves: int = OCTAVES_DEFAULT,
    lacunarity: float = LACUNARITY_DEFAULT,
    persistence: float = PERSISTENCE_DEFAULT,
    noise_amp: float = NOISE_AMP_DEFAULT,
) -> np.ndarray:
    """float32 (NY, NX), B(i,j) = [j,i]; monotone non-decreasing along +i for each row after cum-max."""
    perlin = Perlin2D(seed)
    cols = np.arange(grid_w, dtype=np.float64)
    rows = np.arange(grid_h, dtype=np.float64)
    ii, jj = np.meshgrid(cols, rows)
    nx_grid = (ii / grid_w) * scale
    ny_grid = (jj / grid_h) * scale
    raw = sample_fbm(perlin, nx_grid, ny_grid, octaves, lacunarity, persistence)
    x_ratio = ii.astype(np.float64) / float(grid_w)
    base = column_base_vec(x_ratio)
    perturbed = base + noise_amp * raw
    mono = np.maximum.accumulate(perturbed, axis=1)
    mono = np.maximum(mono, SEA_Z0)
    return mono.astype(np.float32)


def write_raw_f32(path: str, height: np.ndarray) -> None:
    flat = np.ascontiguousarray(height, dtype=np.float32).ravel(order="C")
    with open(path, "wb") as f:
        f.write(flat.tobytes())


def write_png_preview(path: str, height: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("PNG preview requires pillow: pip install pillow") from e
    h = height.astype(np.float64)
    lo, hi = float(h.min()), float(h.max())
    if hi <= lo:
        g = np.zeros_like(h, dtype=np.uint8)
    else:
        g = ((h - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Coastal Perlin+fBm heightmap (raw float32)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nx", type=int, default=GRID_W_DEFAULT, help="columns, C++ NX")
    ap.add_argument("--ny", type=int, default=GRID_H_DEFAULT, help="rows, C++ NY")
    ap.add_argument("--scale", type=float, default=SCALE_DEFAULT)
    ap.add_argument("--octaves", type=int, default=OCTAVES_DEFAULT)
    ap.add_argument("--lacunarity", type=float, default=LACUNARITY_DEFAULT)
    ap.add_argument("--persistence", type=float, default=PERSISTENCE_DEFAULT)
    ap.add_argument(
        "--noise-amp",
        type=float,
        default=NOISE_AMP_DEFAULT,
        help="Perlin fBm scale before monotone cum-max along +x",
    )
    ap.add_argument("--amplitude", type=float, default=1.0, help="global height scale")
    ap.add_argument("-o", "--output", type=str, default="terrain_coastal.raw", help="output .raw path")
    ap.add_argument("--png", type=str, default="", help="optional preview PNG path")
    args = ap.parse_args()

    hmap = generate_terrain(
        args.seed,
        args.nx,
        args.ny,
        scale=args.scale,
        octaves=args.octaves,
        lacunarity=args.lacunarity,
        persistence=args.persistence,
        noise_amp=args.noise_amp,
    )
    if args.amplitude != 1.0:
        hmap = (hmap * np.float32(args.amplitude)).astype(np.float32)

    write_raw_f32(args.output, hmap)
    print(f"Wrote {args.output}  ({args.nx}x{args.ny} float32, {args.nx * args.ny * 4} bytes)")

    if args.png:
        write_png_preview(args.png, hmap)
        print(f"Wrote preview {args.png}")

    hdr_path = args.output.replace(".raw", ".meta.txt")
    try:
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(f"NX={args.nx}\nNY={args.ny}\n")
            f.write(f"dtype=float32\norder=C row-major j then i: index = i + j*NX\n")
            f.write(f"seed={args.seed}\n")
        print(f"Wrote {hdr_path}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
