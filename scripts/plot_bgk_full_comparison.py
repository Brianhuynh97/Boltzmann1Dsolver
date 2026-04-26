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

    full_velocity = load_profile(full_dir / "bulk_vx.txt")

    x = normalized_axis(bgk_temperature.size)
    full_temperature_i = interpolate_to(x, full_temperature)
    full_velocity_i = interpolate_to(x, full_velocity)

    bgk_velocity_label = "BGK"
    full_velocity_label = "full Boltzmann"

    heat_warning = None

    if case_name == "heat_conduction":
        first_panel = (
            "Temperature",
            bgk_temperature,
            full_temperature_i,
            "temperature",
        )
    elif case_name == "poiseuille":
        bgk_velocity_label = "BGK ($u_x$)"
        full_velocity_label = "full Boltzmann ($u_x$)"
        first_panel = (
            "Poiseuille velocity",
            normalized(bgk_velocity),
            normalized(full_velocity_i),
            "flow-direction velocity / max",
        )
    else:
        bgk_velocity_label = "BGK ($u_x$)"
        full_velocity_label = "full Boltzmann ($u_x$)"
        first_panel = (
            "Couette flow-direction velocity shape",
            normalized(bgk_velocity),
            normalized(full_velocity_i),
            "flow-direction velocity / max",
        )

    # Keep the comparison plot available even when the full-Boltzmann heat
    # conduction temperature level looks suspicious; that warning is still
    # useful diagnostic context.
    if case_name == "heat_conduction":
        full_temp_mean = np.mean(full_temperature)
        if abs(full_temp_mean - 2.0) < 0.5:
            heat_warning = (
                f"Warning: Full Boltzmann heat conduction data appears invalid "
                f"(mean temperature {full_temp_mean:.2f} vs expected ~0.8)"
            )
            print(heat_warning)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    panels = [
        first_panel,
        (
            "Temperature variation shape",
            normalized(centered(bgk_temperature)),
            normalized(centered(full_temperature_i)),
            "(T - T_center) / max",
        ),
    ]

    for ax, (title, bgk_values, full_values, xlabel) in zip(axes, panels):
        mismatch = mismatch_linf(bgk_values, full_values)
        if "velocity" in title.lower():
            bgk_label = bgk_velocity_label
            full_label = full_velocity_label
        else:
            bgk_label = "BGK"
            full_label = "full Boltzmann"

        ax.plot(bgk_values, x, label=bgk_label, linewidth=2.0)
        ax.plot(full_values, x, "--", label=full_label, linewidth=2.0)
        ax.set_title(f"{title}\nmax mismatch = {mismatch:.3g}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("normalized channel coordinate")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    if heat_warning is None:
        fig.suptitle("Qualitative BGK/full-Boltzmann comparison", fontsize=11)
    else:
        fig.suptitle(
            "Qualitative BGK/full-Boltzmann comparison\n"
            "full-Boltzmann heat-conduction temperature level looks inconsistent",
            fontsize=11,
        )

    out = full_dir / "bgk_full_comparison.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
