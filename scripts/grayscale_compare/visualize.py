"""Visualization helpers for preprocessing comparison sheets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_float_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    plt.imsave(path, arr, cmap="gray", vmin=0.0, vmax=1.0)


def comparison_grid(original: np.ndarray, outputs: dict[str, np.ndarray], path: Path) -> None:
    names = ["Original"] + list(outputs.keys())
    images = [original] + list(outputs.values())
    cols = min(4, len(images))
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name, img in zip(axes.ravel(), names, images):
        ax.imshow(img, cmap="gray", vmin=0.0 if name != "Original" else None, vmax=1.0 if name != "Original" else None)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def histogram_grid(outputs: dict[str, np.ndarray], path: Path) -> None:
    cols = 3
    rows = int(np.ceil(len(outputs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, img) in zip(axes.ravel(), outputs.items()):
        arr = np.asarray(img, dtype=np.float32).ravel()
        ax.hist(arr, bins=64, range=(0.0, 1.0), color="black", alpha=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlim(0, 1)
        ax.axis("on")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def difference_grid(outputs: dict[str, np.ndarray], reference_name: str, path: Path) -> None:
    reference = outputs[reference_name]
    diffs = {name: np.abs(img - reference) for name, img in outputs.items() if name != reference_name}
    cols = 3
    rows = int(np.ceil(len(diffs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, diff) in zip(axes.ravel(), diffs.items()):
        ax.imshow(diff, cmap="magma", vmin=0.0, vmax=max(float(diff.max()), 1e-6))
        ax.set_title(f"|{name} - {reference_name}|", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
