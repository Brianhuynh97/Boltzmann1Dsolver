#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_profile(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        return data
    return data[-1]


def normalized_axis(size: int) -> np.ndarray:
    if size <= 1:
        return np.zeros(size)
    return np.linspace(0.0, 1.0, size)


def interpolate_to(reference_x: np.ndarray, values: np.ndarray) -> np.ndarray:
    source_x = normalized_axis(values.size)
    return np.interp(reference_x, source_x, values)


def centered(values: np.ndarray) -> np.ndarray:
    return values - values[values.size // 2]


def normalized(values: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(values))
    if scale <= 1e-14:
        return values
    return values / scale


def mismatch_linf(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs - rhs)))


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: python3 plot_bgk_full_comparison.py <case> <bgk_output_dir> <full_output_dir>",
            file=sys.stderr,
        )
        return 1

    case_name = sys.argv[1].lower()
    bgk_dir = Path(sys.argv[2])
    full_dir = Path(sys.argv[3])

    bgk_temperature = load_profile(bgk_dir / "temperature.txt")
    bgk_velocity = load_profile(bgk_dir / "bulk_vx.txt")

    full_temperature = load_profile(full_dir / "temperature.txt")

    full_velocity = load_profile(full_dir / "bulk_vy.txt")

    x = normalized_axis(bgk_temperature.size)
    full_temperature_i = interpolate_to(x, full_temperature)
    full_velocity_i = interpolate_to(x, full_velocity)

    if case_name == "heat_conduction":
        first_panel = (
            "Temperature profile shape",
            normalized(bgk_temperature),
            normalized(full_temperature_i),
            "T / max|T|",
        )
    elif case_name == "poiseuille":
        first_panel = (
            "Poiseuille velocity shape",
            normalized(bgk_velocity),
            normalized(full_velocity_i),
            "stream velocity / max",
        )
    else:
        first_panel = (
            "Couette velocity shape",
            normalized(bgk_velocity),
            normalized(full_velocity_i),
            "stream velocity / max",
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    panels = [
        first_panel,
        ("Temperature variation shape", normalized(centered(bgk_temperature)), normalized(centered(full_temperature_i)), "(T - T_center) / max"),
    ]

    for ax, (title, bgk_values, full_values, xlabel) in zip(axes, panels):
        mismatch = mismatch_linf(bgk_values, full_values)
        ax.plot(bgk_values, x, label="BGK", linewidth=2.0)
        ax.plot(full_values, x, "--", label="full Boltzmann", linewidth=2.0)
        ax.set_title(f"{title}\nmax mismatch = {mismatch:.3g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalized channel coordinate")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle("Qualitative BGK/full-Boltzmann comparison, not calibrated validation", fontsize=11)

    out = full_dir / "bgk_full_comparison.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
