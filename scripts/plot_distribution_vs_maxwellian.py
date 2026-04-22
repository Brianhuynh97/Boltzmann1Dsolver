#!/usr/bin/env python3

from __future__ import annotations

from math import pi
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def load_last_valid_row(path: Path) -> np.ndarray:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    rows = [[float(x) for x in line.split()] for line in lines]
    lengths = [len(row) for row in rows]
    target_len = max(set(lengths), key=lengths.count)

    for row in reversed(rows):
        if len(row) == target_len:
            return np.array(row, dtype=float)

    raise ValueError(f"could not find a valid row in {path}")


def local_maxwellian(vx: np.ndarray, bulk_vx: float, density: float, temperature: float) -> np.ndarray:
    prefactor = density / ((pi * temperature) ** 1.5)
    return prefactor * np.exp(-((vx - bulk_vx) ** 2) / temperature)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_distribution_vs_maxwellian.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    axis = np.loadtxt(output_dir / "velocity_axis.txt")
    density = load_last_valid_row(output_dir / "density.txt")
    bulk_vx = load_last_valid_row(output_dir / "bulk_vx.txt")
    bulk_vy_path = output_dir / "bulk_vy.txt"
    bulk_vy = load_last_valid_row(bulk_vy_path) if bulk_vy_path.exists() else np.zeros_like(density)
    temperature = load_last_valid_row(output_dir / "temperature.txt")

    left = load_matrix(output_dir / "distribution_left.txt")
    center = load_matrix(output_dir / "distribution_center.txt")

    iy0 = len(axis) // 2
    left_line = left[iy0, :]
    center_line = center[iy0, :]

    # Saved slices may be either raw f or perturbation f - f_eq. Detect by sign.
    is_perturbation = min(float(left.min()), float(center.min())) < 0.0

    left_eq = local_maxwellian(axis, bulk_vx[0], density[0], temperature[0])
    center_index = len(density) // 2
    center_eq = local_maxwellian(axis, bulk_vx[center_index], density[center_index], temperature[center_index])

    left_actual = left_line + left_eq if is_perturbation else left_line
    center_actual = center_line + center_eq if is_perturbation else center_line

    left_delta = left_actual - left_eq
    center_delta = center_actual - center_eq

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True, sharex=True)

    axes[0, 0].plot(axis, left_actual, label="Distribution", color="tab:blue", linewidth=2)
    axes[0, 0].plot(axis, left_eq, label="Maxwellian", color="tab:orange", linewidth=2, linestyle="--")
    axes[0, 0].set_title("Wall")
    axes[0, 0].set_ylabel(r"$f(v_x, v_y=0, v_z=0)$")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(axis, center_actual, label="Distribution", color="tab:blue", linewidth=2)
    axes[0, 1].plot(axis, center_eq, label="Maxwellian", color="tab:orange", linewidth=2, linestyle="--")
    axes[0, 1].set_title("Center")
    axes[0, 1].set_ylabel(r"$f(v_x, v_y=0, v_z=0)$")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(axis, left_delta, color="tab:red", linewidth=2)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 0].set_title("Wall deviation from Maxwellian")
    axes[1, 0].set_xlabel(r"$v_x$")
    axes[1, 0].set_ylabel(r"$f - f_{eq}$")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(axis, center_delta, color="tab:red", linewidth=2)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_title("Center deviation from Maxwellian")
    axes[1, 1].set_xlabel(r"$v_x$")
    axes[1, 1].set_ylabel(r"$f - f_{eq}$")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Wall/Center Distribution Compared with Local Maxwellian")

    out = output_dir / "distribution_vs_maxwellian.png"
    fig.savefig(out, dpi=180)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
