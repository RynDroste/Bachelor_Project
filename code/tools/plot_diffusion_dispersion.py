#!/usr/bin/env python3
"""
读取 diffusion_dispersion_bench 生成的 CSV，绘制
  x: Wavelength (m)
  y: Relative wave speed
四条曲线对比扩散迭代 8 / 16 / 32 / 128。

优先使用 matplotlib 生成 PNG；若未安装则生成 SVG（无第三方依赖）。

用法:
  ./diffusion_dispersion_bench > build/diffusion_dispersion.csv
  python3 tools/plot_diffusion_dispersion.py build/diffusion_dispersion.csv -o build/diffusion_dispersion.png
  python3 tools/plot_diffusion_dispersion.py build/diffusion_dispersion.csv -o build/diffusion_dispersion.svg
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> dict[int, list[tuple[float, float]]]:
    series: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            niter = int(row["niter"])
            lam = float(row["wavelength_m"])
            rel = float(row["relative_wave_speed"])
            series[niter].append((lam, rel))
    for k in series:
        series[k].sort(key=lambda t: t[0])
    return dict(series)


def plot_matplotlib(data: dict[int, list[tuple[float, float]]], out: Path) -> None:
    import matplotlib.pyplot as plt

    niters = [8, 16, 32, 128]
    colors = ["#c0392b", "#7f8c8d", "#e67e22", "#2980b9"]
    plt.figure(figsize=(8.5, 5.0))
    for n, c in zip(niters, colors):
        if n not in data:
            continue
        xs = [p[0] for p in data[n]]
        ys = [p[1] for p in data[n]]
        plt.plot(
            xs,
            ys,
            color=c,
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=f"{n} diffusion iterations",
        )

    plt.xlabel("Wavelength (m)")
    plt.ylabel("Relative wave speed")
    plt.title("Effect of wave decomposition (diffusion iterations) on effective dispersion")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper right")
    plt.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.45)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_svg(data: dict[int, list[tuple[float, float]]], out: Path) -> None:
    """无 matplotlib 时用 SVG 折线（同一坐标系）。"""
    niters = [8, 16, 32, 128]
    colors = ["#c0392b", "#7f8c8d", "#e67e22", "#2980b9"]
    W, H = 900, 520
    margin_l, margin_r, margin_t, margin_b = 72, 40, 48, 56
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    all_pts = [p for n in niters if n in data for p in data[n]]
    if not all_pts:
        raise SystemExit("CSV 中无数据")
    xs_all = [p[0] for p in all_pts]
    ys_all = [p[1] for p in all_pts]
    x0, x1 = min(xs_all), max(xs_all)
    y0 = min(min(ys_all), 0.95)
    y1 = max(max(ys_all), 1.05)
    if abs(x1 - x0) < 1e-9:
        x1 = x0 + 1.0
    if abs(y1 - y0) < 1e-9:
        y1 = y0 + 0.1

    def tx(x: float) -> float:
        return margin_l + (x - x0) / (x1 - x0) * plot_w

    def ty(y: float) -> float:
        return margin_t + (y1 - y) / (y1 - y0) * plot_h

    lines: list[str] = []
    for n, col in zip(niters, colors):
        if n not in data:
            continue
        pts = data[n]
        d = "M " + " L ".join(f"{tx(p[0]):.2f},{ty(p[1]):.2f}" for p in pts)
        lines.append(f'<path fill="none" stroke="{col}" stroke-width="2.2" d="{d}"/>')
        for p in pts:
            cx, cy = tx(p[0]), ty(p[1])
            lines.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" fill="{col}"/>')

    # axes
    x_axis_y = ty(1.0)
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" '
        f'y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" '
        f'y2="{margin_t + plot_h}" stroke="#333" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{margin_l}" y1="{x_axis_y:.2f}" x2="{margin_l + plot_w}" '
        f'y2="{x_axis_y:.2f}" stroke="#888" stroke-width="1" stroke-dasharray="6,4"/>'
    )

    # ticks x
    for i in range(0, 9):
        lam = x0 + (x1 - x0) * i / 8.0
        x_ = tx(lam)
        lines.append(
            f'<line x1="{x_:.2f}" y1="{margin_t + plot_h}" x2="{x_:.2f}" '
            f'y2="{margin_t + plot_h + 5}" stroke="#333"/>'
        )
        lines.append(
            f'<text x="{x_:.2f}" y="{H - 18}" text-anchor="middle" font-size="13" '
            f'font-family="sans-serif">{lam:.1f}</text>'
        )

    # ticks y
    for i in range(0, 7):
        v = y0 + (y1 - y0) * i / 6.0
        y_ = ty(v)
        lines.append(
            f'<line x1="{margin_l - 5}" y1="{y_:.2f}" x2="{margin_l}" '
            f'y2="{y_:.2f}" stroke="#333"/>'
        )
        lines.append(
            f'<text x="{margin_l - 10}" y="{y_ + 4:.2f}" text-anchor="end" font-size="13" '
            f'font-family="sans-serif">{v:.2f}</text>'
        )

    lines.append(
        f'<text x="{W // 2}" y="{H - 4}" text-anchor="middle" font-size="15" '
        f'font-family="sans-serif">Wavelength (m)</text>'
    )
    lines.append(
        f'<text transform="rotate(-90 {22} {margin_t + plot_h // 2})" x="22" '
        f'y="{margin_t + plot_h // 2}" text-anchor="middle" font-size="15" '
        f'font-family="sans-serif">Relative wave speed</text>'
    )

    leg_x, leg_y = margin_l + plot_w - 8, margin_t + 12
    for i, (n, col) in enumerate(zip(niters, colors)):
        if n not in data:
            continue
        ly = leg_y + i * 22
        lines.append(f'<rect x="{leg_x - 150}" y="{ly - 10}" width="14" height="14" fill="{col}"/>')
        lines.append(
            f'<text x="{leg_x - 130}" y="{ly + 2}" font-size="13" font-family="sans-serif">'
            f"{n} diffusion iterations</text>"
        )

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        *lines,
        "</svg>",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot relative wave speed vs wavelength for diffusion iterations.")
    ap.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("build/diffusion_dispersion.csv"),
        help="CSV from diffusion_dispersion_bench",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image (.png needs matplotlib, .svg always works)",
    )
    ap.add_argument("--svg", action="store_true", help="Force SVG output (no matplotlib)")
    args = ap.parse_args()

    data = load_csv(args.csv)
    out = args.output
    if out is None:
        out = args.csv.with_suffix(".png")

    if args.svg or out.suffix.lower() == ".svg":
        svg_path = out if out.suffix.lower() == ".svg" else out.with_suffix(".svg")
        plot_svg(data, svg_path)
        print(f"Wrote {svg_path} (SVG, no matplotlib)")
        return

    try:
        plot_matplotlib(data, out)
        print(f"Wrote {out}")
    except ImportError:
        svg_path = out.with_suffix(".svg")
        plot_svg(data, svg_path)
        print(f"matplotlib 未安装，已改为 SVG: {svg_path}")


if __name__ == "__main__":
    main()
