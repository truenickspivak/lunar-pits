"""Inspect generated LROC NAC tile manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def create_contact_sheet(df: pd.DataFrame, output_path: Path, sample_size: int, seed: int) -> None:
    saved = df[df["saved"] & df["tile_path"].astype(bool)]
    if saved.empty:
        print("No saved tiles available for contact sheet.")
        return

    sample = saved.sample(n=min(sample_size, len(saved)), random_state=seed)
    cols = min(5, len(sample))
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        with rasterio.open(row["tile_path"]) as src:
            arr = src.read(1, masked=True)
        values = arr.compressed()
        if values.size:
            p2, p98 = np.percentile(values, [2, 98])
            shown = np.clip((arr.filled(p2) - p2) / max(p98 - p2, 1e-12), 0, 1)
        else:
            shown = np.zeros(arr.shape)
        ax.imshow(shown, cmap="gray")
        ax.set_title(row["tile_id"], fontsize=8)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"contact sheet: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect LROC NAC tile manifest.")
    parser.add_argument("--parquet", default="data/tiles/nac/nac_tiles.parquet")
    parser.add_argument("--contact-sheet", default=None, help="Optional output PNG path")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    df = pd.read_parquet(parquet_path)
    print(f"manifest: {parquet_path}")
    print("status counts:")
    print(df["status"].value_counts(dropna=False))
    print("\nsaved tile examples:")
    print(df.loc[df["saved"], "tile_path"].head(10).to_string(index=False))

    if args.contact_sheet:
        create_contact_sheet(df, Path(args.contact_sheet), args.sample_size, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

