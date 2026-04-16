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
        print("usage: python3 plot_distribution_contour.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    axis = np.loadtxt(output_dir / "velocity_axis.txt")
    time_text = (output_dir / "distribution_time.txt").read_text(encoding="utf-8").strip()
    try:
        time_label = f"t = {float(time_text):.4f}"
    except ValueError:
        time_label = time_text

    left = load_matrix(output_dir / "distribution_left.txt")
    center = load_matrix(output_dir / "distribution_center.txt")
    right = load_matrix(output_dir / "distribution_right.txt")

    labels_path = output_dir / "distribution_labels.txt"
    if labels_path.exists():
        labels = labels_path.read_text(encoding="utf-8").splitlines()
        while len(labels) < 3:
            labels.append("")
    else:
        labels = ["Left wall", "Center", "Right wall"]

    vx, vy = np.meshgrid(axis, axis)
    fields = [
        (labels[0] or "Left wall", left),
        (labels[1] or "Center", center),
        (labels[2] or "Right wall", right),
    ]

    vmin = min(float(field.min()) for _, field in fields)
    vmax = max(float(field.max()) for _, field in fields)
    amplitude = max(abs(vmin), abs(vmax))
    if np.isclose(amplitude, 0.0):
        amplitude = 1e-12

    levels = np.linspace(-amplitude, amplitude, 15)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)

    for ax, (title, field) in zip(axes, fields):
        contour = ax.contourf(vx, vy, field, levels=levels, cmap="coolwarm", extend="both")
        ax.contour(vx, vy, field, levels=levels, colors="black", linewidths=0.5, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel("v_x")
        ax.set_ylabel("v_y")
        ax.set_aspect("equal")

    fig.colorbar(contour, ax=axes, shrink=0.88, label="f(v_x, v_y, v_z = 0) - f_0")
    fig.suptitle(f"Velocity-Space Distribution Slices at {time_label}")

    out = output_dir / "distribution_contour.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
