"""Export PNG quicklooks for selected LROC tile manifest rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def save_tile_quicklooks(
    parquet_path: str | Path = "data/tiles/nac/nac_tiles.parquet",
    out_dir: str | Path = "data/tiles/nac/interesting_quicklooks",
    status: str = "interesting_kept",
    max_tiles: int | None = None,
) -> int:
    parquet_path = Path(parquet_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    rows = df[df["saved"] & df["tile_path"].astype(bool) & df["status"].eq(status)].copy()
    if max_tiles is not None:
        rows = rows.head(max_tiles)

    for _, row in rows.iterrows():
        with rasterio.open(row["tile_path"]) as src:
            arr = src.read(1, masked=True)
        values = arr.compressed()
        if values.size:
            p2, p98 = np.percentile(values, [2, 98])
            shown = np.clip((arr.filled(p2) - p2) / max(p98 - p2, 1e-12), 0, 1)
        else:
            shown = np.zeros(arr.shape, dtype=np.float32)
        plt.imsave(out_dir / f"{row['tile_id']}.png", shown, cmap="gray")

    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PNG quicklooks for saved LROC tiles.")
    parser.add_argument("--parquet", default="data/tiles/nac/nac_tiles.parquet")
    parser.add_argument("--out-dir", default="data/tiles/nac/interesting_quicklooks")
    parser.add_argument("--status", default="interesting_kept")
    parser.add_argument("--max-tiles", type=int, default=None)
    args = parser.parse_args()

    count = save_tile_quicklooks(args.parquet, args.out_dir, args.status, args.max_tiles)
    print(f"exported {count} {args.status} quicklooks to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
