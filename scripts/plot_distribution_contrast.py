#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_distribution_contrast.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    axis = np.loadtxt(output_dir / "velocity_axis.txt")

    left = load_matrix(output_dir / "distribution_left.txt")
    center = load_matrix(output_dir / "distribution_center.txt")
    right = load_matrix(output_dir / "distribution_right.txt")

    left_diff = left - center
    right_diff = right - center

    amplitude = max(
        float(np.max(np.abs(left_diff))),
        float(np.max(np.abs(right_diff))),
        1e-12,
    )

    levels = np.linspace(-amplitude, amplitude, 17)
    vx, vy = np.meshgrid(axis, axis)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    fields = [
        ("Left wall - Center", left_diff),
        ("Right wall - Center", right_diff),
    ]

    for ax, (title, field) in zip(axes, fields):
        contour = ax.contourf(vx, vy, field, levels=levels, cmap="coolwarm", extend="both")
        ax.contour(vx, vy, field, levels=levels, colors="black", linewidths=0.45, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("v_x")
        ax.set_ylabel("v_y")
        ax.set_aspect("equal")

    fig.colorbar(contour, ax=axes, shrink=0.9, label="difference from center slice")
    fig.suptitle("Distribution Contrast Relative to Channel Center")

    out = output_dir / "distribution_contrast.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
