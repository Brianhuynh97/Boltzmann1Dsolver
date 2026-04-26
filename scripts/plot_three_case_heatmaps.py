#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_1d(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim != 1:
        raise ValueError(f"expected 1D data in {path}, got shape {data.shape}")
    return data


def load_last_row(path: Path) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        return data
    return data[-1]


def tile_profile(profile: np.ndarray, width: int = 220) -> np.ndarray:
    return np.repeat(profile[:, np.newaxis], width, axis=1)


def main() -> int:
    couette_dir = ROOT / "output/bgk_channel/couette"
    poiseuille_dir = ROOT / "output/bgk_channel/poiseuille"
    heat_dir = ROOT / "output/bgk_channel/heat_conduction"

    couette_profile = load_1d(couette_dir / "bulk_vx.txt")
    poiseuille_profile = load_1d(poiseuille_dir / "bulk_vx.txt")
    heat_profile = load_last_row(heat_dir / "temperature.txt")

    panels = [
        ("(a) Couette flow: streamwise velocity $u_x$", tile_profile(couette_profile), "turbo"),
        ("(b) Poiseuille flow: driven streamwise velocity $u_x$", tile_profile(poiseuille_profile), "turbo"),
        ("(c) Heat conduction: temperature $T$", tile_profile(heat_profile), "turbo"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), constrained_layout=True)

    for ax, (title, image, cmap) in zip(axes, panels):
        mesh = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap, interpolation="bicubic")
        ax.set_ylabel("wall-normal cell index")
        ax.set_title(title, fontsize=11, pad=8)
        fig.colorbar(mesh, ax=ax, orientation="vertical", pad=0.012, shrink=0.92)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.85")

    out = ROOT / "output/three_case_heatmaps.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
