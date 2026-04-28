#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def load_matrix(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 animate_distribution_x_v.py <output_dir>", file=sys.stderr)
        return 1

    output_dir = Path(sys.argv[1])
    snapshot_dir = output_dir / "snapshots"
    x_path = output_dir / "x_cells.txt"
    v_path = output_dir / "velocity_axis.txt"
    times_path = snapshot_dir / "snapshot_times.txt"

    if not snapshot_dir.exists() or not x_path.exists() or not v_path.exists() or not times_path.exists():
        print("Missing required files for animation.", file=sys.stderr)
        return 1

    frame_paths = sorted(snapshot_dir.glob("distribution_f_*.txt"))
    if not frame_paths:
        print("No snapshot frames found.", file=sys.stderr)
        return 1

    x = np.loadtxt(x_path)
    v = np.loadtxt(v_path)
    times = np.loadtxt(times_path)
    if times.ndim == 1:
        times = times[np.newaxis, :]

    frames = [load_matrix(path) for path in frame_paths]
    sample_count = min(6, len(x))
    sample_indices = np.linspace(0, len(x) - 1, sample_count, dtype=int)
    sample_indices = np.unique(sample_indices)
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(sample_indices)))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    def draw_frame(frame_idx: int) -> None:
        ax.clear()
        frame = frames[frame_idx]
        for color, sample_index in zip(colors, sample_indices):
            x0 = float(x[sample_index])
            profile = frame[sample_index, :]
            ax.plot(
                v,
                profile,
                color=color,
                linewidth=2.0,
                label=f"x={x0:.2f}",
                zorder=2,
            )

        time_value = times[min(frame_idx, len(times) - 1), 1]
        ax.set_title(f"Distribution f(c) at Selected x, t={time_value:.4f}")
        ax.set_xlabel("velocity c")
        ax.set_ylabel("distribution f")
        ax.set_xlim(float(v[0]), float(v[-1]))
        frame_peak = max(float(np.max(frame)), 1e-12)
        ax.set_ylim(0.0, 1.05 * frame_peak)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    draw_frame(0)
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames), interval=260, repeat=True)

    out = output_dir / "distribution_f.gif"
    try:
        anim.save(out, writer=animation.PillowWriter(fps=4))
    except Exception as exc:
        print(f"failed to save GIF: {exc}", file=sys.stderr)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
