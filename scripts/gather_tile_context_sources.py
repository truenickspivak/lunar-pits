"""Gather LOLA/GRAIL/Diviner context for one deterministic lunar tile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lunar_tile_pipeline.tiling import get_tile_for_latlon
from lunarpits.location.context_sources import (
    load_context_config,
    sample_diviner_context_for_tile,
    sample_grail_context_for_tile,
    sample_lola_context_for_tile,
    write_json,
)
from lunarpits.location.gather_location import location_output_dir_name
from lunarpits.location.gather_location import site_tile_output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample coarse context sources for a deterministic lunar tile.")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--tile-size-km", type=float, default=0.256)
    parser.add_argument("--site", default=None, help="Optional clean site label for data/locations/<site>/...")
    parser.add_argument("--context-dir", default=None)
    parser.add_argument("--config", default=str(ROOT / "config" / "context_sources.yaml"))
    parser.add_argument("--download", action="store_true", help="Download/cache configured coarse source files before sampling.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    tile = get_tile_for_latlon(args.lat, args.lon, tile_size_km=args.tile_size_km)
    site_name = args.site or location_output_dir_name(args.lat, args.lon)
    out_dir = Path(args.context_dir) if args.context_dir else site_tile_output_dir(site_name, args.tile_size_km)
    cache_dir = ROOT / "data" / "cache" / "context_sources"
    config = load_context_config(args.config)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "tile.json", tile.to_metadata())
    topology = sample_lola_context_for_tile(tile, config.get("lola", {}), cache_dir=cache_dir, download=args.download)
    gravity = sample_grail_context_for_tile(tile, config.get("grail", {}), cache_dir=cache_dir, download=args.download)
    ir = sample_diviner_context_for_tile(tile, config.get("diviner", {}), cache_dir=cache_dir, download=args.download)
    write_json(out_dir / "topology" / "context.json", topology)
    write_json(out_dir / "gravity" / "context.json", gravity)
    write_json(out_dir / "ir" / "context.json", ir)

    print(f"tile: {tile.tile_id}")
    print(f"context folder: {out_dir}")
    print(f"topology/LOLA available: {topology.get('available')}")
    print(f"gravity/GRAIL available: {gravity.get('available')}")
    print(f"IR/Diviner available: {ir.get('available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
