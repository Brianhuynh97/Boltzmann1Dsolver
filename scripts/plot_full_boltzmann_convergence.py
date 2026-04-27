#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "output/full_boltzmann_1d3v"


def load_residual_history(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    history = np.loadtxt(case_dir / "convergence_history.txt")
    if history.ndim == 1:
        history = history.reshape(1, -1)

    iterations = history[:, 0]
    residual = np.maximum(history[:, 1], 1e-16)
    return iterations, residual


def main() -> int:
    cases = [
        ("Couette", OUTPUT_ROOT / "couette"),
        ("Poiseuille", OUTPUT_ROOT / "poiseuille"),
        ("Heat conduction", OUTPUT_ROOT / "heat_conduction"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)

    for label, case_dir in cases:
        if not case_dir.exists():
            continue
        iterations, residual = load_residual_history(case_dir)
        if residual.size == 0:
            continue
        ax.semilogy(iterations, residual, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("iteration")
    ax.set_ylabel("relative residual")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    fig.suptitle("Full-Boltzmann Convergence", fontsize=12)

    out = OUTPUT_ROOT / "full_boltzmann_convergence.png"
    fig.savefig(out, dpi=160)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
