#!/usr/bin/env python3
"""
不考虑离散网格：用连续介质近似 wave decompose 中的扩散阶段。

假定平坦水深 h0、恒定扩散系数 α = h0²/64（与 GPU 上 grad→0 时的 alpha 一致），
每步等效时间 ΔT = 0.25 s（与 wave_decompose_gpu.cu 中 dt 上限一致）。
初始为单一长波空间模式，波数 k = 2π/λ，λ = 2π h0。

傅里叶振幅每步乘以 exp(-α k² ΔT)，经 n 步后保留因子 R(n) = exp(-α k² ΔT n)。
迭代次数 n：0 以及 2^k（k=0..7）。无第三方依赖。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

H0 = 4.0  # m
DT_STEP = 0.25  # 与 GPU 扩散子步时间上限一致
TWO_PI = 2.0 * math.pi

# 长波：λ = 2π h0 ⇒ k = 1/h0
WAVELENGTH_M = TWO_PI * H0


def alpha_flat(h0: float) -> float:
    return h0 * h0 / 64.0


def wavenumber_from_lambda(wavelength_m: float) -> float:
    return TWO_PI / wavelength_m


def retention_after_iters(n_iter: int, h0: float, wavelength_m: float, dt_step: float) -> float:
    if n_iter <= 0:
        return 1.0
    alpha = alpha_flat(h0)
    k = wavenumber_from_lambda(wavelength_m)
    return math.exp(-alpha * k * k * dt_step * float(n_iter))


def iteration_values() -> list[int]:
    vals: list[int] = [0]
    for kk in range(8):
        vals.append(1 << kk)
    return vals


def fmt3(x: float) -> str:
    """CSV 中浮点数字符串，固定三位小数。"""
    return f"{float(x):.3f}"


def main() -> None:
    out_path = Path(__file__).resolve().parent / "wave_decompose_long_wave_decay.csv"
    rows: list[dict[str, str | int]] = []

    for n_iter in iteration_values():
        r = retention_after_iters(n_iter, H0, WAVELENGTH_M, DT_STEP)
        decay = 1.0 - r

        rows.append(
            {
                "n": n_iter,
                "retention_amplitude_ratio": fmt3(r),
                "decay_ratio_amplitude": fmt3(decay),
            }
        )

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
