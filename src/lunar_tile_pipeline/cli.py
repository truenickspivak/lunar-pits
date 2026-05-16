"""Command line interface for deterministic lunar tile processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lunar_tile_pipeline.config import DEFAULT_OUTPUT_DIR
from lunar_tile_pipeline.diviner import sample_dataset_for_tile as sample_diviner_for_tile
from lunar_tile_pipeline.grail import sample_dataset_for_tile as sample_grail_for_tile
from lunar_tile_pipeline.io import tile_output_dir, write_products, write_tile_metadata
from lunar_tile_pipeline.lola import sample_dataset_for_tile as sample_lola_for_tile
from lunar_tile_pipeline.lroc import enrich_lroc_metadata, find_lroc_nac_edr_for_tile_from_file, rank_lroc_nac_edr_for_tile
from lunar_tile_pipeline.tiling import (
    get_tile_by_indices,
    get_tile_for_latlon,
    iter_global_tiles,
    iter_tiles_for_latlon_bounds,
    parse_tile_id,
)


def process_tile(tile, args, *, input_query: dict[str, object] | None = None) -> Path:
    folder = tile_output_dir(tile, args.out)
    data_sources = {
        "lroc_footprints": args.footprints,
        "grail": getattr(args, "grail", None),
        "diviner": getattr(args, "diviner", None),
        "lola": getattr(args, "lola", None),
    }
    write_tile_metadata(tile, args.out, input_query=input_query, data_sources=data_sources)
    if args.footprints:
        products = find_lroc_nac_edr_for_tile_from_file(tile, args.footprints)
    else:
        products = pd.DataFrame()
    if getattr(args, "enrich", False):
        products = enrich_lroc_metadata(products)
    products = rank_lroc_nac_edr_for_tile(products)
    write_products(tile, args.out, products)
    for subdir in ("nac_crops", "grail", "diviner", "lola", "masks"):
        (folder / subdir).mkdir(exist_ok=True)
    from lunar_tile_pipeline.io import write_json

    write_json(folder / "grail" / "grail_context.json", sample_grail_for_tile(tile, getattr(args, "grail", None), folder / "grail"))
    write_json(folder / "diviner" / "diviner_context.json", sample_diviner_for_tile(tile, getattr(args, "diviner", None), folder / "diviner"))
    write_json(folder / "lola" / "lola_context.json", sample_lola_for_tile(tile, getattr(args, "lola", None), folder / "lola"))
    labels = folder / "labels.json"
    if not labels.exists():
        labels.write_text("[]\n", encoding="utf-8")
    return folder


def cmd_process_point(args: argparse.Namespace) -> int:
    tile = get_tile_for_latlon(args.lat, args.lon, tile_size_km=args.tile_size_km)
    folder = process_tile(tile, args, input_query={"lat": args.lat, "lon_original": args.lon})
    print(folder)
    return 0


def cmd_process_region(args: argparse.Namespace) -> int:
    count = 0
    for tile in iter_tiles_for_latlon_bounds(args.min_lat, args.max_lat, args.min_lon, args.max_lon, args.tile_size_km):
        process_tile(tile, args)
        count += 1
    print(f"processed tiles: {count}")
    return 0


def cmd_list_global(args: argparse.Namespace) -> int:
    rows = []
    for tile in iter_global_tiles(args.tile_size_km):
        x_min, y_min, x_max, y_max = tile.bounds_xy
        rows.append(
            {
                "tile_id": tile.tile_id,
                "tile_x": tile.tile_x,
                "tile_y": tile.tile_y,
                "center_lat": tile.center_lat,
                "center_lon_180": tile.center_lon_180,
                "center_lon_360": tile.center_lon_360,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)
    return 0


def cmd_inspect_tile(args: argparse.Namespace) -> int:
    tile_size_km, tile_x, tile_y = parse_tile_id(args.tile_id)
    tile = get_tile_by_indices(tile_x, tile_y, tile_size_km=tile_size_km)
    print(tile.to_metadata())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lunar-tiles", description="Deterministic lunar tile-grid pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    point = sub.add_parser("process-point")
    point.add_argument("--lat", type=float, required=True)
    point.add_argument("--lon", type=float, required=True)
    point.add_argument("--tile-size-km", type=float, default=10)
    point.add_argument("--footprints", default=None)
    point.add_argument("--grail", default=None)
    point.add_argument("--diviner", default=None)
    point.add_argument("--lola", default=None)
    point.add_argument("--enrich", dest="enrich", action="store_true", default=False)
    point.add_argument("--no-enrich", dest="enrich", action="store_false")
    point.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    point.set_defaults(func=cmd_process_point)

    region = sub.add_parser("process-region")
    region.add_argument("--min-lat", type=float, required=True)
    region.add_argument("--max-lat", type=float, required=True)
    region.add_argument("--min-lon", type=float, required=True)
    region.add_argument("--max-lon", type=float, required=True)
    region.add_argument("--tile-size-km", type=float, default=10)
    region.add_argument("--footprints", default=None)
    region.add_argument("--grail", default=None)
    region.add_argument("--diviner", default=None)
    region.add_argument("--lola", default=None)
    region.add_argument("--enrich", dest="enrich", action="store_true", default=False)
    region.add_argument("--no-enrich", dest="enrich", action="store_false")
    region.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    region.set_defaults(func=cmd_process_region)

    global_cmd = sub.add_parser("list-global")
    global_cmd.add_argument("--tile-size-km", type=float, default=10)
    global_cmd.add_argument("--out", required=True)
    global_cmd.set_defaults(func=cmd_list_global)

    inspect = sub.add_parser("inspect-tile")
    inspect.add_argument("--tile-id", required=True)
    inspect.set_defaults(func=cmd_inspect_tile)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
