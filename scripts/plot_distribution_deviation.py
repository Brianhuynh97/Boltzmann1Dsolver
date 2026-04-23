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

    rows = [[float(value) for value in line.split()] for line in lines]
    lengths = [len(row) for row in rows]
    target_len = max(set(lengths), key=lengths.count)

    for row in reversed(rows):
        if len(row) == target_len:
            return np.array(row, dtype=float)

    raise ValueError(f"could not find a valid row in {path}")


def local_maxwellian_2d(
    vx: np.ndarray,
    vy: np.ndarray,
    vz_value: float,
    density: float,
    bulk_vx: float,
    bulk_vy: float,
    temperature: float,
) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-12)
    prefactor = density / ((pi * safe_temperature) ** 1.5)
    shifted_speed_sq = (vx - bulk_vx) ** 2 + (vy - bulk_vy) ** 2 + vz_value**2
    return prefactor * np.exp(-shifted_speed_sq / safe_temperature)


def resolve_slice(matrix: np.ndarray, eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if float(matrix.min()) < 0.0:
        actual = matrix + eq
    else:
        actual = matrix
    return actual, actual - eq


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_distribution_deviation.py <output_dir>", file=sys.stderr)
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

    if left.shape != center.shape:
        raise ValueError("left and center distribution slices must have the same shape")

    iy0 = len(axis) // 2
    iz0 = len(axis) // 2
    vz_value = float(axis[iz0])

    vx_grid, vy_grid = np.meshgrid(axis, axis)
    left_eq = local_maxwellian_2d(
        vx_grid,
        vy_grid,
        vz_value,
        float(density[0]),
        float(bulk_vx[0]),
        float(bulk_vy[0]),
        float(temperature[0]),
    )
    center_index = len(density) // 2
    center_eq = local_maxwellian_2d(
        vx_grid,
        vy_grid,
        vz_value,
        float(density[center_index]),
        float(bulk_vx[center_index]),
        float(bulk_vy[center_index]),
        float(temperature[center_index]),
    )

    left_actual, left_delta = resolve_slice(left, left_eq)
    center_actual, center_delta = resolve_slice(center, center_eq)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)

    left_line = left_delta[iy0, :]
    center_line = center_delta[iy0, :]
    line_amplitude = max(float(np.max(np.abs(left_line))), float(np.max(np.abs(center_line))), 1e-12)

    for ax, title, values in (
        (axes[0], "Wall Slice Deviation", left_line),
        (axes[1], "Center Slice Deviation", center_line),
    ):
        ax.plot(axis, values, color="tab:red", linewidth=2.2)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.fill_between(axis, 0.0, values, where=values >= 0.0, color="tab:red", alpha=0.22)
        ax.fill_between(axis, 0.0, values, where=values <= 0.0, color="tab:blue", alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel(r"$v_x$")
        ax.set_ylabel(r"$f - f_{eq}$ at $v_y \approx 0$")
        ax.set_ylim(-1.1 * line_amplitude, 1.1 * line_amplitude)
        ax.grid(True, alpha=0.28)

    fig.suptitle("Distribution Deviation from Local Maxwellian")

    out = output_dir / "distribution_deviation.png"
    fig.savefig(out, dpi=180)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
