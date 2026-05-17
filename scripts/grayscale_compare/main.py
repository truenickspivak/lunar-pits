"""Run grayscale preprocessing comparisons on lunar skylight imagery.

Expected dataset layout:

    dataset/
      skylights/
      non_skylights/

You can also pass one or more --input-glob patterns to compare existing tiles.
Outputs include processed images, side-by-side sheets, histograms, difference
maps, and a metrics CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from metrics import summarize
from preprocessing import build_methods, compute_percentile_constants
from visualize import comparison_grid, difference_grid, histogram_grid, save_float_png


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read grayscale image: {path}")
    return img


def collect_images(dataset_dir: Path, input_globs: list[str]) -> list[Path]:
    paths: list[Path] = []
    if input_globs:
        for pattern in input_globs:
            paths.extend(Path().glob(pattern))
    else:
        for subdir in (dataset_dir / "skylights", dataset_dir / "non_skylights"):
            if subdir.exists():
                paths.extend(path for path in subdir.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
    unique = sorted(set(path.resolve() for path in paths if path.suffix.lower() in SUPPORTED_SUFFIXES))
    if not unique:
        raise RuntimeError("No input images found. Provide dataset/skylights + dataset/non_skylights or --input-glob.")
    return unique


def sample_for_percentiles(paths: list[Path], sample_size: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    chosen = paths if len(paths) <= sample_size else rng.sample(paths, sample_size)
    return [load_grayscale(path) for path in chosen]


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def run(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    paths = collect_images(Path(args.dataset), args.input_glob)
    if args.max_images:
        paths = paths[: args.max_images]

    constants = compute_percentile_constants(sample_for_percentiles(paths, args.percentile_sample_size, args.seed))
    methods = build_methods(constants)

    out_dir = Path(args.out_dir)
    processed_dir = out_dir / "processed"
    grids_dir = out_dir / "comparison_grids"
    hist_dir = out_dir / "histograms"
    diff_dir = out_dir / "difference_maps"
    metrics_rows: list[dict[str, object]] = []

    for path in paths:
        original = load_grayscale(path)
        outputs = {name: method(original) for name, method in methods.items()}
        image_id = safe_stem(path)

        for name, img in outputs.items():
            save_float_png(processed_dir / name / f"{image_id}.png", img)
            row = {"image": str(path), "image_id": image_id, "method": name}
            row.update(summarize(img))
            metrics_rows.append(row)

        comparison_grid(original, outputs, grids_dir / f"{image_id}_comparison.png")
        histogram_grid(outputs, hist_dir / f"{image_id}_histograms.png")
        difference_grid(outputs, "A_raw_global_8_245", diff_dir / f"{image_id}_difference_from_A.png")

    metrics_path = out_dir / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    constants_path = out_dir / "percentile_constants.txt"
    constants_path.write_text(
        "\n".join(
            [
                f"seed={args.seed}",
                f"raw_p2={constants.raw_p2}",
                f"raw_p98={constants.raw_p98}",
                f"clahe_p2={constants.clahe_p2}",
                f"clahe_p98={constants.clahe_p98}",
                "",
                "Note: B_per_image_minmax is useful for visual debugging but can remove illumination consistency.",
                "Fixed/global policies preserve shadow brightness relationships across observations.",
                "CLAHE can reveal terrain texture but may alter local shadow morphology.",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare grayscale preprocessing methods for lunar skylight imagery.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--input-glob", action="append", default=[], help="Optional glob pattern; can be repeated.")
    parser.add_argument("--out-dir", default="data/diagnostics/grayscale_framework")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--percentile-sample-size", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = run(args)
    print(f"Wrote grayscale comparison outputs to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
