#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_linearized_bgk_channel.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    y = np.loadtxt(output_dir / "y_cells.txt")
    density = np.loadtxt(output_dir / "density.txt")
    bulk_vx = np.loadtxt(output_dir / "bulk_vx.txt")
    temperature = np.loadtxt(output_dir / "temperature.txt")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(bulk_vx, y, color="tab:blue", linewidth=2)
    axes[0].set_title("Couette velocity")
    axes[0].set_xlabel("u_x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(density, y, color="tab:green", linewidth=2)
    axes[1].set_title("Density")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("y")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(temperature, y, color="tab:red", linewidth=2)
    axes[2].set_title("Temperature")
    axes[2].set_xlabel("T")
    axes[2].set_ylabel("y")
    axes[2].grid(True, alpha=0.3)

    out = output_dir / "channel_profile.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
