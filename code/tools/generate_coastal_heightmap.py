#!/usr/bin/env python3
"""
Flat riverbed: every cell has the same elevation (bed height).

Outputs
-------
1) .raw  little-endian float32, NX*NY values, C row-major matching C++ Grid::terrain:
      index = i + j * NX   (i = column/x, j = row/y)
      rows j=0..NY-1, each row i=0..NX-1.
2) Optional .png preview (needs pillow): grayscale (a flat map → uniform mid-grey).

Requires numpy; for --png also: pip install pillow

Examples:
  python generate_coastal_heightmap.py -o ../assets/terrain_flat_128.raw
  python generate_coastal_heightmap.py --nx 256 --ny 256 --bed 0.0 --png preview.png
  python generate_coastal_heightmap.py --bed 1.5 --amplitude 0.5
"""

from __future__ import annotations

import argparse

import numpy as np

GRID_W_DEFAULT = 256
GRID_H_DEFAULT = 256

BED_HEIGHT_DEFAULT = 0.0


def generate_terrain(grid_w: int, grid_h: int, bed_height: float = BED_HEIGHT_DEFAULT) -> np.ndarray:
    """float32 (NY, NX) filled with a single constant elevation."""
    return np.full((grid_h, grid_w), float(bed_height), dtype=np.float32)


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
        # A flat field has no contrast; emit uniform mid-grey.
        g = np.full(h.shape, 128, dtype=np.uint8)
    else:
        g = ((h - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(g, mode="L").save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Flat riverbed heightmap (raw float32)")
    ap.add_argument("--nx", type=int, default=GRID_W_DEFAULT, help="columns, C++ NX")
    ap.add_argument("--ny", type=int, default=GRID_H_DEFAULT, help="rows, C++ NY")
    ap.add_argument(
        "--bed",
        type=float,
        default=BED_HEIGHT_DEFAULT,
        help="constant bed elevation for every cell (default 0.0)",
    )
    ap.add_argument("--amplitude", type=float, default=1.0, help="global height scale (applied after --bed)")
    ap.add_argument("-o", "--output", type=str, default="terrain_flat.raw", help="output .raw path")
    ap.add_argument("--png", type=str, default="", help="optional preview PNG path")
    args = ap.parse_args()

    hmap = generate_terrain(args.nx, args.ny, bed_height=args.bed)
    if args.amplitude != 1.0:
        hmap = (hmap * np.float32(args.amplitude)).astype(np.float32)

    write_raw_f32(args.output, hmap)
    print(
        f"Wrote {args.output}  ({args.nx}x{args.ny} float32, "
        f"{args.nx * args.ny * 4} bytes, bed={args.bed * args.amplitude:.6g})"
    )

    if args.png:
        write_png_preview(args.png, hmap)
        print(f"Wrote preview {args.png}")

    hdr_path = args.output.replace(".raw", ".meta.txt")
    try:
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(f"NX={args.nx}\nNY={args.ny}\n")
            f.write("dtype=float32\norder=C row-major j then i: index = i + j*NX\n")
            f.write(f"bed={args.bed * args.amplitude:.6g}\n")
        print(f"Wrote {hdr_path}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
