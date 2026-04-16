from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        return data[None, :]
    return data


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_macro_heatmap.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])

    t = np.loadtxt(output_dir / "t_levels.txt")
    density = load_series(output_dir / "density.txt")
    bulk_vx = load_series(output_dir / "bulk_vx.txt")
    bulk_vy = load_series(output_dir / "bulk_vy.txt")
    temperature = load_series(output_dir / "temperature.txt")

    if np.ndim(t) == 0:
        t = np.array([float(t)])

    x = np.arange(density.shape[1] + 1)
    t_edges = np.linspace(t[0], t[-1], len(t) + 1) if len(t) > 1 else np.array([t[0], t[0] + 1.0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fields = [
        ("Density", density, "rho"),
        ("Bulk velocity x", bulk_vx, "u_x"),
        ("Bulk velocity y", bulk_vy, "u_y"),
        ("Temperature", temperature, "T"),
    ]

    for ax, (title, values, label) in zip(axes.flat, fields):
        mesh = ax.pcolormesh(x, t_edges, values, shading="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Cell index")
        ax.set_ylabel("time")
        fig.colorbar(mesh, ax=ax, label=label)

    out = output_dir / "heatmap_plot.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
