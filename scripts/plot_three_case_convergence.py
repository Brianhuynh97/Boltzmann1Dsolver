#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output" / "bgk_channel"


def load_channel_residual(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    history = np.loadtxt(case_dir / "convergence_history.txt")
    if history.ndim == 1:
        history = history.reshape(1, -1)

    iterations = history[:, 0]
    residual = np.maximum(history[:, 1], 1e-16)
    return iterations, residual


def main() -> int:
    cases = [
        (
            "Couette",
            *load_channel_residual(OUTPUT_ROOT / "couette"),
            "iteration",
            "relative residual",
        ),
        (
            "Poiseuille",
            *load_channel_residual(OUTPUT_ROOT / "poiseuille"),
            "iteration",
            "relative residual",
        ),
        (
            "Heat conduction",
            *load_channel_residual(OUTPUT_ROOT / "heat_conduction"),
            "iteration",
            "relative residual",
        ),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.5), constrained_layout=True)

    for ax, (title, x_values, errors, x_label, y_label) in zip(axes, cases):
        ax.semilogy(x_values, errors, marker="o", linewidth=1.8, markersize=3.8)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)

    out = ROOT / "output" / "three_case_convergence.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
