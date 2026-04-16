from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        return data[None, :]
    return data


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 plot_full_boltzmann_shock_tube.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    t = np.loadtxt(output_dir / "t_levels.txt")
    density = _load_series(output_dir / "density.txt")
    bulk_vx = _load_series(output_dir / "bulk_vx.txt")
    bulk_vy = _load_series(output_dir / "bulk_vy.txt")
    temperature = _load_series(output_dir / "temperature.txt")

    if density.size == 0 or bulk_vx.size == 0 or bulk_vy.size == 0 or temperature.size == 0 or np.size(t) == 0:
        print(f"no data available in {output_dir}", file=sys.stderr)
        return 1

    if np.ndim(t) == 0:
        t = np.array([float(t)])

    x = np.arange(density.shape[1])

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)

    axes[0].plot(x, density[0], "--", label=f"t={t[0]:.4f}")
    axes[0].plot(x, density[-1], label=f"t={t[-1]:.4f}")
    axes[0].set_title("Density")
    axes[0].set_xlabel("Cell index")
    axes[0].set_ylabel("rho")
    axes[0].legend()

    axes[1].plot(x, bulk_vx[0], "--", label=f"t={t[0]:.4f}")
    axes[1].plot(x, bulk_vx[-1], label=f"t={t[-1]:.4f}")
    axes[1].set_title("Bulk velocity in x")
    axes[1].set_xlabel("Cell index")
    axes[1].set_ylabel("u_x")
    axes[1].legend()

    axes[2].plot(x, bulk_vy[0], "--", label=f"t={t[0]:.4f}")
    axes[2].plot(x, bulk_vy[-1], label=f"t={t[-1]:.4f}")
    axes[2].set_title("Bulk velocity in y")
    axes[2].set_xlabel("Cell index")
    axes[2].set_ylabel("u_y")
    axes[2].legend()

    axes[3].plot(x, temperature[0], "--", label=f"t={t[0]:.4f}")
    axes[3].plot(x, temperature[-1], label=f"t={t[-1]:.4f}")
    axes[3].set_title("Temperature")
    axes[3].set_xlabel("Cell index")
    axes[3].set_ylabel("T")
    axes[3].legend()

    plot_path = output_dir / "summary_plot.png"
    fig.savefig(plot_path, dpi=160)
    print(plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
