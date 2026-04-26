#!/usr/bin/env python3

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


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_bgk_channel.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    case_name = output_dir.name.lower()
    axis_path = output_dir / "y_cells.txt"
    density = load_profile(output_dir / "density.txt")
    bulk_vx = load_profile(output_dir / "bulk_vx.txt")
    temperature = load_profile(output_dir / "temperature.txt")
    wall_temperature_path = output_dir / "wall_temperature.txt"
    wall_temperature = load_profile(wall_temperature_path) if wall_temperature_path.exists() else None

    if axis_path.exists():
        coord = np.loadtxt(axis_path)
        coord_label = "y"
    else:
        coord = np.arange(density.size)
        coord_label = "Cell index"

    if case_name == "poiseuille":
        primary_velocity = bulk_vx
        velocity_label = "u_x"
        velocity_title = "Poiseuille velocity"
    elif case_name == "heat_conduction":
        primary_velocity = bulk_vx
        velocity_label = "u_x"
        velocity_title = "Velocity (near zero)"
    elif case_name == "couette":
        primary_velocity = bulk_vx
        velocity_label = "u_x"
        velocity_title = "Couette velocity"
    else:
        primary_velocity = bulk_vx
        velocity_label = "u_x"
        velocity_title = "Velocity"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(primary_velocity, coord, color="tab:blue", linewidth=2)
    axes[0].set_title(velocity_title)
    axes[0].set_xlabel(velocity_label)
    axes[0].set_ylabel(coord_label)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(density, coord, color="tab:green", linewidth=2)
    axes[1].set_title("Density")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel(coord_label)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temperature, coord, color="tab:red", linewidth=2)
    if wall_temperature is not None and wall_temperature.size >= 2:
        axes[2].scatter(
            [wall_temperature[0], wall_temperature[1]],
            [coord[0], coord[-1]],
            color="black",
            s=28,
            zorder=3,
            label="wall T",
        )
        axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Temperature")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel(coord_label)
    axes[2].grid(True, alpha=0.3)

    out = output_dir / "channel_profile.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
