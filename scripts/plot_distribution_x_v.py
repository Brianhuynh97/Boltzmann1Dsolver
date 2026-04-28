#!/usr/bin/env python3

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

def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_distribution_x_v.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    x_cells_path = output_dir / "x_cells.txt"
    velocity_axis_path = output_dir / "velocity_axis.txt"
    matrix_path = output_dir / "distribution_f.txt"
    bulk_velocity_path = output_dir / "bulk_vx.txt"

    if not velocity_axis_path.exists() or not matrix_path.exists():
        print("Missing required output files. Expected velocity_axis.txt and distribution_f.txt.", file=sys.stderr)
        return 1

    if x_cells_path.exists():
        x_cells = np.loadtxt(x_cells_path)
    else:
        x_cells = None

    axis = np.loadtxt(velocity_axis_path)
    distribution = load_matrix(matrix_path)

    if x_cells is None or distribution.shape[0] != x_cells.size:
        x_cells = np.arange(distribution.shape[0], dtype=float)

    v_grid, x_grid = np.meshgrid(axis, x_cells)

    fig3d = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax3d = fig3d.add_subplot(111, projection="3d")
    surface = ax3d.plot_surface(
        x_grid,
        v_grid,
        distribution,
        cmap="plasma",
        linewidth=0.15,
        edgecolor=(0, 0, 0, 0.08),
        antialiased=True,
        rcount=min(120, distribution.shape[0]),
        ccount=min(120, distribution.shape[1]),
        shade=True,
        alpha=0.96,
    )
    colorbar3d = fig3d.colorbar(surface, ax=ax3d, shrink=0.72, pad=0.08)
    colorbar3d.set_label("distribution f")
    ax3d.contour(
        x_grid,
        v_grid,
        distribution,
        zdir="z",
        offset=0.0,
        levels=12,
        cmap="plasma",
        linewidths=0.8,
        alpha=0.9,
    )
    ax3d.set_title("Distribution f(x, c)")
    ax3d.set_xlabel("spatial coordinate x")
    ax3d.set_ylabel("velocity c")
    ax3d.set_zlabel("distribution f")
    ax3d.set_zlim(0.0, float(np.max(distribution)) * 1.02)
    ax3d.view_init(elev=18, azim=0)
    out3d = output_dir / "distribution_f.png"
    fig3d.savefig(out3d, dpi=160)

    if bulk_velocity_path.exists():
        bulk_velocity = np.loadtxt(bulk_velocity_path)
    else:
        dv = axis[1] - axis[0] if axis.size > 1 else 1.0
        bulk_velocity = np.sum(distribution * axis[np.newaxis, :], axis=1) * dv / np.maximum(
            np.sum(distribution, axis=1) * dv,
            1e-12,
        )

    fig2d, ax2d = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax2d.plot(
        x_cells,
        bulk_velocity,
        color="black",
        linewidth=2.0,
    )
    ax2d.set_title("Velocity c Along x")
    ax2d.set_xlabel("spatial coordinate x")
    ax2d.set_ylabel("velocity c")
    ax2d.grid(True, alpha=0.25)
    out2d = output_dir / "velocity_c_x.png"
    fig2d.savefig(out2d, dpi=180)

    print(out3d)
    print(out2d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
