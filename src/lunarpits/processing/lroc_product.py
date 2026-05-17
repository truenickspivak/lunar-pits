"""
Process one LROC NAC EDR product from IMG ID to final GeoTIFF.

Windows-side controller:
- Finds the IMG URL in the LROC PDS EDR archive collection CSV files.
- Downloads only the IMG to a temporary WSL-native workspace.
- Runs ISIS inside WSL/Ubuntu.
- Uses spiceinit web=yes to avoid local full-LRO ISISDATA download.
- Copies only the final TIFF back to the Windows project folder.
- Deletes temporary IMG/CUB files by default.
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from lunarpits.processing.identifiers import normalize_product_id


PROJECT_WIN = Path(r"C:\Users\nicks\desktop\lunar-pits")
PDS_ROOT = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-2-EDR-V1.0"
WSL_WORK_ROOT = "/tmp/lunar-pits-work"
WSL_CACHE_ROOT = "/tmp/lunar-pits-cache"
ISIS_ENV_NAME = "isis9.0.0"
OUT_DIR_WIN = PROJECT_WIN / "data" / "processed" / "lroc_tif"
TILE_OUT_DIR_WIN = PROJECT_WIN / "data" / "processed" / "lroc_tile_tif"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocatedProduct:
    product_id: str
    volume: str
    relative_path: str
    img_url: str
    xml_url: str | None = None


@dataclass(frozen=True)
class LrocProcessingPlan:
    product_id: str
    final_tif_win: Path
    final_tif_wsl: str
    work_dir: str
    pixel_resolution: float
    center_lat: float | None
    center_lon: float | None
    tile_map: "TileMapBounds | None" = None
    shape: str | None = None
    shape_model: str | None = None
    echo_correct: bool = False
    spksmithed: bool = False


@dataclass(frozen=True)
class TileMapBounds:
    tile_id: str
    center_lat: float
    center_lon: float
    projection_center_lat: float
    projection_center_lon: float
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass(frozen=True)
class LrocProcessingResult:
    product_id: str
    final_tif: Path
    skipped: bool = False
    dry_run: bool = False

    def __fspath__(self) -> str:
        return str(self.final_tif)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def win_to_wsl(path: Path) -> str:
    """Convert C:\\foo\\bar to /mnt/c/foo/bar."""
    path = path.resolve()
    drive = path.drive[0].lower()
    rest = str(path)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def output_tif_path(product_id: str, out_dir: Path = OUT_DIR_WIN) -> Path:
    product_id = normalize_product_id(product_id)
    return out_dir / f"{product_id}.map.tif"


def output_tile_tif_path(product_id: str, tile_id: str) -> Path:
    product_id = normalize_product_id(product_id)
    return TILE_OUT_DIR_WIN / tile_id / f"{product_id}.map.tif"


def geotiff_mask_path(tif_path: Path) -> Path:
    return Path(str(tif_path) + ".msk")


def has_existing_geotiff_output(tif_path: Path) -> bool:
    """Return true when a prior GDAL GeoTIFF export appears complete."""
    return (
        tif_path.exists()
        and tif_path.stat().st_size > 0
        and geotiff_mask_path(tif_path).exists()
        and geotiff_mask_path(tif_path).stat().st_size > 0
    )


def build_processing_plan(
    product_id: str,
    *,
    center_lat: float | None,
    center_lon: float | None,
    pixel_resolution: float = 0.5,
    tile_map: TileMapBounds | None = None,
    output_dir: Path | None = None,
    shape: str | None = None,
    shape_model: str | None = None,
    echo_correct: bool = False,
    spksmithed: bool = False,
) -> LrocProcessingPlan:
    product_id = normalize_product_id(product_id)
    if output_dir is not None:
        final_tif_win = Path(output_dir) / f"{product_id}.map.tif"
    else:
        final_tif_win = output_tile_tif_path(product_id, tile_map.tile_id) if tile_map else output_tif_path(product_id)
    return LrocProcessingPlan(
        product_id=product_id,
        final_tif_win=final_tif_win,
        final_tif_wsl=win_to_wsl(final_tif_win),
        work_dir=f"{WSL_WORK_ROOT}/{product_id}",
        pixel_resolution=pixel_resolution,
        center_lat=tile_map.center_lat if tile_map else center_lat,
        center_lon=tile_map.center_lon if tile_map else center_lon,
        tile_map=tile_map,
        shape=shape,
        shape_model=shape_model,
        echo_correct=echo_correct,
        spksmithed=spksmithed,
    )


def run_wsl_shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a normal shell command inside WSL."""
    LOGGER.info("\n[WSL] %s\n", command)
    return subprocess.run(
        ["wsl", "bash", "-lc", command],
        text=True,
        check=check,
    )


def run_wsl_shell_capture(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a normal shell command inside WSL and capture output."""
    LOGGER.debug("\n[WSL capture] %s\n", command)
    return subprocess.run(
        ["wsl", "bash", "-lc", command],
        text=True,
        check=check,
        capture_output=True,
    )


def run_isis(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ISIS command inside the configured WSL conda environment."""
    full_cmd = f"~/miniforge3/bin/conda run -n {ISIS_ENV_NAME} {command}"
    LOGGER.info("\n[ISIS] %s\n", command)
    return subprocess.run(
        ["wsl", "bash", "-lc", full_cmd],
        text=True,
        check=check,
    )


def run_isis_capture(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ISIS command inside WSL and capture output."""
    full_cmd = f"~/miniforge3/bin/conda run -n {ISIS_ENV_NAME} {command}"
    LOGGER.debug("\n[ISIS capture] %s\n", command)
    return subprocess.run(
        ["wsl", "bash", "-lc", full_cmd],
        text=True,
        check=check,
        capture_output=True,
    )


def build_spiceinit_command(
    cub_wsl: str,
    *,
    shape: str | None = None,
    shape_model: str | None = None,
    spksmithed: bool = False,
    require_smithed: bool = False,
) -> str:
    """Build spiceinit command while preserving legacy behavior by default."""
    if shape is None and not spksmithed and not require_smithed:
        return f"spiceinit from='{cub_wsl}' web=yes"

    parts = [f"spiceinit from='{cub_wsl}'", "web=true"]
    if spksmithed or require_smithed:
        parts.append("spksmithed=true")
    if require_smithed:
        parts.append("spkrecon=false")
    if shape is not None:
        normalized = shape.lower()
        if normalized not in {"system", "ellipsoid", "user"}:
            raise ValueError("shape must be one of: system, ellipsoid, user")
        if normalized == "user":
            if not shape_model:
                raise ValueError("shape=user requires --shape-model")
            parts.extend(["shape=user", f"model='{shape_model}'"])
        else:
            if shape_model:
                raise ValueError("--shape-model can only be used with shape=user")
            parts.append(f"shape={normalized}")
    elif shape_model:
        raise ValueError("--shape-model requires shape=user")
    return " ".join(parts)


def timed_step(label: str, func: Callable[..., object], *args: object, **kwargs: object) -> object:
    start = time.perf_counter()
    LOGGER.info("[start] %s", label)
    try:
        return func(*args, **kwargs)
    finally:
        LOGGER.info("[time] %s %.1fs", label, time.perf_counter() - start)


def read_url_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download_url_to_wsl(url: str, dest_wsl: str) -> None:
    run_wsl_shell(f"mkdir -p $(dirname '{dest_wsl}') && wget -q --show-progress -O '{dest_wsl}' '{url}'")


def ensure_final_dir(path: Path = OUT_DIR_WIN) -> None:
    path.mkdir(parents=True, exist_ok=True)


def likely_volume_names(max_volume: int = 80) -> Iterable[str]:
    """
    Generate likely LROC volume names.

    The archive includes names like:
        LROLRC_0001
        ...
        LROLRC_0060C
    """
    for i in range(1, max_volume + 1):
        yield f"LROLRC_{i:04d}"
        for suffix in ("A", "B", "C", "D"):
            yield f"LROLRC_{i:04d}{suffix}"


def collection_csv_url(volume: str) -> str:
    lower = volume.lower()
    return f"{PDS_ROOT}/{volume}/DATA/collection_lro-l-lroc-2-edr_{lower}_data.csv"


def extract_img_path_from_collection_row(row_text: str, product_id: str) -> str | None:
    """
    Try to pull a relative IMG path from a collection CSV row.

    Expected path-like target examples:
        MAP/2009258/NAC/M107689129RE.IMG
        SCI/2011308/NAC/M175057326RE.IMG
        DATA/MAP/2009258/NAC/M107689129RE.IMG
        ./MAP/2009258/NAC/M107689129RE.IMG
    """
    product_img = f"{product_id}.IMG"
    pattern = re.compile(rf"(?:DATA/)?(?:MAP|COM|SCI)/[^,\"']*?/NAC/{re.escape(product_img)}", re.IGNORECASE)
    match = pattern.search(row_text)
    if match:
        path = match.group(0).replace("\\", "/")
        path = path.removeprefix("./")
        return path

    if product_img in row_text.upper():
        return product_img

    return None


def locate_product_in_collections(product_id: str, *, max_volume: int = 80) -> LocatedProduct:
    product_id = normalize_product_id(product_id)

    LOGGER.info("[locate] Searching PDS collection CSVs for %s.IMG", product_id)

    for volume in likely_volume_names(max_volume=max_volume):
        csv_url = collection_csv_url(volume)
        try:
            text = read_url_text(csv_url, timeout=20)
        except Exception:
            continue

        if product_id not in text.upper():
            LOGGER.debug("[skip] %s", volume)
            continue

        LOGGER.info("[hit] %s appears in %s", product_id, volume)

        for line in text.splitlines():
            if product_id in line.upper():
                rel = extract_img_path_from_collection_row(line, product_id)
                if rel is None:
                    continue

                if rel.upper().endswith(".IMG") and "/" in rel:
                    rel = rel.removeprefix("DATA/")
                    img_url = f"{PDS_ROOT}/{volume}/DATA/{rel}"
                    xml_url = img_url[:-4] + ".xml"
                    return LocatedProduct(product_id, volume, rel, img_url, xml_url)

                break

        found = brute_force_volume_for_product(product_id, volume)
        if found:
            return found

        raise RuntimeError(
            f"Found {product_id} in {volume}'s collection CSV, but could not resolve its IMG path."
        )

    raise RuntimeError(f"Could not locate {product_id}.IMG in searched LROC EDR collection CSVs.")


def parse_href_links(html: str) -> list[str]:
    return re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)


def brute_force_volume_for_product(product_id: str, volume: str) -> LocatedProduct | None:
    """Fallback HTML directory scraper for one known volume."""
    product_img = f"{product_id}.IMG"

    for branch in ("MAP", "COM", "SCI"):
        branch_url = f"{PDS_ROOT}/{volume}/DATA/{branch}/"
        try:
            branch_html = read_url_text(branch_url, timeout=20)
        except Exception:
            continue

        day_dirs = [h for h in parse_href_links(branch_html) if h.strip("/").isdigit()]
        for day in day_dirs:
            day = day.strip("/")
            nac_url = f"{branch_url}{day}/NAC/"
            try:
                nac_html = read_url_text(nac_url, timeout=20)
            except Exception:
                continue

            if product_img in nac_html:
                rel = f"{branch}/{day}/NAC/{product_img}"
                img_url = f"{PDS_ROOT}/{volume}/DATA/{rel}"
                xml_url = img_url[:-4] + ".xml"
                return LocatedProduct(product_id, volume, rel, img_url, xml_url)

    return None


def parse_caminfo_center(caminfo_text: str) -> tuple[float, float]:
    """Extract approximate footprint center from ISIS caminfo PVL text."""

    def find_number(key: str, *, required: bool = True) -> float | None:
        match = re.search(rf"\b{key}\s*=\s*([-+]?\d+(?:\.\d+)?)", caminfo_text)
        if not match:
            if not required:
                return None
            raise RuntimeError(f"Could not find {key} in caminfo output.")
        return float(match.group(1))

    center_lat = find_number("CenterLatitude", required=False)
    center_lon = find_number("CenterLongitude", required=False)
    if center_lat is not None and center_lon is not None:
        return center_lat, center_lon

    min_lat = find_number("MinimumLatitude")
    max_lat = find_number("MaximumLatitude")
    min_lon = find_number("MinimumLongitude")
    max_lon = find_number("MaximumLongitude")

    center_lat = (min_lat + max_lat) / 2.0

    if max_lon - min_lon > 180.0:
        center_lon = ((min_lon + max_lon + 360.0) / 2.0) % 360.0
    else:
        center_lon = (min_lon + max_lon) / 2.0

    return center_lat, center_lon


def tile_map_bounds_from_latlon(lat: float, lon: float, tile_size_km: float) -> TileMapBounds:
    """Build ISIS map bounds for the deterministic lunar tile containing lat/lon."""
    from lunar_tile_pipeline.tiling import get_tile_for_latlon

    tile = get_tile_for_latlon(lat, lon, tile_size_km=tile_size_km)
    bounds = tile.bounds_latlon
    if bool(bounds.get("crosses_lon_360_wrap")):
        raise ValueError(f"Tile {tile.tile_id} crosses the 0/360 longitude seam; fixed-tile cam2map seam handling is TODO.")
    return TileMapBounds(
        tile_id=tile.tile_id,
        center_lat=float(tile.center_lat),
        center_lon=float(tile.center_lon_360),
        projection_center_lat=0.0,
        projection_center_lon=0.0,
        min_lat=float(bounds["min_lat_approx"]),
        max_lat=float(bounds["max_lat_approx"]),
        min_lon=float(bounds["min_lon_360_approx"]),
        max_lon=float(bounds["max_lon_360_approx"]),
    )


def write_map_file_wsl(
    map_path_wsl: str,
    center_lat: float,
    center_lon: float,
    pixel_resolution: float,
    tile_map: TileMapBounds | None = None,
) -> None:
    range_text = ""
    if tile_map is not None:
        range_text = f"""  MinimumLatitude    = {tile_map.min_lat}
  MaximumLatitude    = {tile_map.max_lat}
  MinimumLongitude   = {tile_map.min_lon}
  MaximumLongitude   = {tile_map.max_lon}
"""
    map_text = f"""Group = Mapping
  ProjectionName     = Equirectangular
  TargetName         = Moon
  EquatorialRadius   = 1737400.0 <meters>
  PolarRadius        = 1737400.0 <meters>
  LatitudeType       = Planetocentric
  LongitudeDirection = PositiveEast
  LongitudeDomain    = 360
  CenterLatitude     = {tile_map.projection_center_lat if tile_map is not None else center_lat}
  CenterLongitude    = {tile_map.projection_center_lon if tile_map is not None else center_lon}
  PixelResolution    = {pixel_resolution} <meters/pixel>
{range_text}End_Group
End
"""

    run_wsl_shell(
        f"mkdir -p $(dirname '{map_path_wsl}') && cat > '{map_path_wsl}' <<'EOF'\n{map_text}EOF"
    )


def process_product(
    product_id: str,
    *,
    center_lat: float | None,
    center_lon: float | None,
    pixel_resolution: float = 0.5,
    tile_map: TileMapBounds | None = None,
    output_dir: Path | None = None,
    img_url: str | None = None,
    keep_temp: bool = False,
    max_volume: int = 80,
    skip_if_exists: bool = False,
    dry_run: bool = False,
    use_isis_cache: bool = True,
    refresh_isis_cache: bool = False,
    shape: str | None = None,
    shape_model: str | None = None,
    spksmithed: bool = False,
    require_smithed: bool = False,
    echo_correct: bool = False,
    forwardpatch: bool = False,
    patch_size: int = 100,
) -> LrocProcessingResult:
    product_id = normalize_product_id(product_id)
    plan = build_processing_plan(
        product_id,
        center_lat=center_lat,
        center_lon=center_lon,
        pixel_resolution=pixel_resolution,
        tile_map=tile_map,
        output_dir=output_dir,
        shape=shape,
        shape_model=shape_model,
        echo_correct=echo_correct,
        spksmithed=spksmithed,
    )

    if skip_if_exists and has_existing_geotiff_output(plan.final_tif_win):
        LOGGER.info("[skip] Existing GeoTIFF found: %s", plan.final_tif_win)
        return LrocProcessingResult(product_id, plan.final_tif_win, skipped=True)
    if skip_if_exists and plan.final_tif_win.exists():
        LOGGER.info("[reprocess] Existing TIFF is missing GDAL GeoTIFF sidecar metadata; rebuilding: %s", plan.final_tif_win)

    LOGGER.info("[plan] product_id=%s", plan.product_id)
    LOGGER.info("[plan] output=%s", plan.final_tif_win)
    LOGGER.info("[plan] pixel_resolution=%s meters/pixel", plan.pixel_resolution)

    if dry_run:
        LOGGER.info("[dry-run] No PDS download, WSL command, or ISIS command will be run.")
        return LrocProcessingResult(product_id, plan.final_tif_win, dry_run=True)

    ensure_final_dir(plan.final_tif_win.parent)

    work_dir = plan.work_dir
    cache_suffix = "default"
    if shape:
        cache_suffix = shape.lower()
        if shape_model:
            cache_suffix = f"{cache_suffix}-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(shape_model).name)
    cache_dir = f"{WSL_CACHE_ROOT}/{product_id}/{cache_suffix}"
    cached_cal_wsl = f"{cache_dir}/{product_id}.cal.cub"
    img_wsl = f"{work_dir}/{product_id}.IMG"
    cub_wsl = f"{work_dir}/{product_id}.cub"
    cal_wsl = f"{work_dir}/{product_id}.cal.cub"
    echo_wsl = f"{work_dir}/{product_id}.echo.cub"
    caminfo_wsl = f"{work_dir}/{product_id}.caminfo.pvl"
    map_pvl_wsl = f"{work_dir}/{product_id}.map.pvl"
    map_cub_wsl = f"{work_dir}/{product_id}.map.cub"
    temp_geotiff_wsl = f"{work_dir}/{product_id}.map.geotiff.tif"

    run_wsl_shell(f"rm -rf '{work_dir}' && mkdir -p '{work_dir}'")
    final_created = False

    try:
        cache_hit = False
        if use_isis_cache and tile_map is not None and not refresh_isis_cache:
            cache_hit = run_wsl_shell(f"test -s '{cached_cal_wsl}'", check=False).returncode == 0

        if cache_hit:
            LOGGER.info("[cache] Reusing calibrated ISIS cube: %s", cached_cal_wsl)
            timed_step("copy cached calibrated cube", run_wsl_shell, f"cp '{cached_cal_wsl}' '{cal_wsl}'")
        else:
            located = (
                LocatedProduct(product_id, "direct_url", "", img_url, None)
                if img_url
                else locate_product_in_collections(product_id, max_volume=max_volume)
            )
            LOGGER.info("[located] %s", located.img_url)
            timed_step("download IMG", download_url_to_wsl, located.img_url, img_wsl)
            timed_step("lronac2isis", run_isis, f"lronac2isis from='{img_wsl}' to='{cub_wsl}'")
            timed_step(
                "spiceinit",
                run_isis,
                build_spiceinit_command(
                    cub_wsl,
                    shape=shape,
                    shape_model=shape_model,
                    spksmithed=spksmithed,
                    require_smithed=require_smithed,
                ),
            )

        if tile_map is not None:
            center_lat = tile_map.center_lat
            center_lon = tile_map.center_lon
            LOGGER.info(
                "[tile map] %s lat %.6f..%.6f lon %.6f..%.6f",
                tile_map.tile_id,
                tile_map.min_lat,
                tile_map.max_lat,
                tile_map.min_lon,
                tile_map.max_lon,
            )
        elif center_lat is None or center_lon is None:
            run_isis(f"caminfo from='{cub_wsl}' to='{caminfo_wsl}'")
            caminfo_text = run_wsl_shell_capture(f"cat '{caminfo_wsl}'").stdout
            inferred_lat, inferred_lon = parse_caminfo_center(caminfo_text)
            center_lat = inferred_lat if center_lat is None else center_lat
            center_lon = inferred_lon if center_lon is None else center_lon
            LOGGER.info("[map center] inferred center_lat=%.6f, center_lon=%.6f", center_lat, center_lon)
        else:
            LOGGER.info("[map center] using override center_lat=%.6f, center_lon=%.6f", center_lat, center_lon)

        effective_pixel_resolution = pixel_resolution
        if abs(center_lat) >= 85.0 and pixel_resolution < 10.0:
            effective_pixel_resolution = 10.0
            LOGGER.info(
                "[map resolution] high-latitude frame detected; using %.1f meters/pixel to avoid oversized polar output",
                effective_pixel_resolution,
            )

        write_map_file_wsl(map_pvl_wsl, center_lat, center_lon, effective_pixel_resolution, tile_map)

        if not cache_hit:
            timed_step("lronaccal", run_isis, f"lronaccal from='{cub_wsl}' to='{cal_wsl}'")
            if use_isis_cache:
                timed_step(
                    "cache calibrated cube",
                    run_wsl_shell,
                    f"mkdir -p '{cache_dir}' && cp '{cal_wsl}' '{cached_cal_wsl}'",
                )
        map_input_wsl = cal_wsl
        if echo_correct:
            timed_step("lronacecho", run_isis, f"lronacecho from='{cal_wsl}' to='{echo_wsl}'")
            map_input_wsl = echo_wsl
        default_range = "map" if tile_map is not None else "camera"
        warp_args = f" warpalgorithm=forwardpatch patchsize={int(patch_size)}" if forwardpatch else ""
        timed_step(
            "cam2map",
            run_isis,
            f"cam2map from='{map_input_wsl}' map='{map_pvl_wsl}' to='{map_cub_wsl}' defaultrange={default_range} pixres=map{warp_args}",
        )
        timed_step(
            "gdal_translate GeoTIFF",
            run_isis,
            "gdal_translate -of GTiff -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER "
            f"'{map_cub_wsl}' '{temp_geotiff_wsl}'",
        )
        run_wsl_shell(f"test -s '{temp_geotiff_wsl}'")

        run_wsl_shell(
            f"mkdir -p $(dirname '{plan.final_tif_wsl}') && "
            f"cp '{temp_geotiff_wsl}' '{plan.final_tif_wsl}' && "
            f"if [ -s '{temp_geotiff_wsl}.msk' ]; then cp '{temp_geotiff_wsl}.msk' '{plan.final_tif_wsl}.msk'; fi"
        )

        if not plan.final_tif_win.exists() or plan.final_tif_win.stat().st_size == 0:
            raise RuntimeError(f"Final TIFF was not created correctly: {plan.final_tif_win}")

        final_created = True
        LOGGER.info("\n[DONE] %s", plan.final_tif_win)
        return LrocProcessingResult(product_id, plan.final_tif_win)

    finally:
        if keep_temp:
            LOGGER.info("[keep-temp] Temporary WSL files left in: %s", work_dir)
        elif not final_created:
            LOGGER.info("[keep-temp] Final GeoTIFF was not completed; temporary WSL files left for inspection: %s", work_dir)
        else:
            LOGGER.info("[cleanup] Removing temporary WSL files: %s", work_dir)
            run_wsl_shell(f"rm -rf '{work_dir}'", check=False)


def process_product_to_geotiff(
    product_id: str,
    *,
    center_lat: float | None = None,
    center_lon: float | None = None,
    pixel_resolution: float = 0.5,
    tile_lat: float | None = None,
    tile_lon: float | None = None,
    tile_size_km: float | None = None,
    output_dir: str | Path | None = None,
    img_url: str | None = None,
    keep_temp: bool = False,
    max_volume: int = 80,
    skip_if_exists: bool = False,
    use_isis_cache: bool = True,
    refresh_isis_cache: bool = False,
    shape: str | None = None,
    shape_model: str | None = None,
    spksmithed: bool = False,
    require_smithed: bool = False,
    echo_correct: bool = False,
    forwardpatch: bool = False,
    patch_size: int = 100,
) -> Path:
    """Process a LROC product id and return the Python-readable GeoTIFF path."""
    tile_map = None
    if tile_lat is not None or tile_lon is not None or tile_size_km is not None:
        if tile_lat is None or tile_lon is None or tile_size_km is None:
            raise ValueError("tile_lat, tile_lon, and tile_size_km must all be provided for fixed-tile processing.")
        tile_map = tile_map_bounds_from_latlon(tile_lat, tile_lon, tile_size_km)
    result = process_product(
        product_id,
        center_lat=center_lat,
        center_lon=center_lon,
        pixel_resolution=pixel_resolution,
        tile_map=tile_map,
        output_dir=Path(output_dir) if output_dir is not None else None,
        img_url=img_url,
        keep_temp=keep_temp,
        max_volume=max_volume,
        skip_if_exists=skip_if_exists,
        use_isis_cache=use_isis_cache,
        refresh_isis_cache=refresh_isis_cache,
        shape=shape,
        shape_model=shape_model,
        spksmithed=spksmithed,
        require_smithed=require_smithed,
        echo_correct=echo_correct,
        forwardpatch=forwardpatch,
        patch_size=patch_size,
    )
    return result.final_tif


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and process one LROC NAC EDR IMG through ISIS.")
    parser.add_argument("product_id", help="Example: M114328462RE")
    parser.add_argument("--center-lat", type=float, default=None, help="Optional map center latitude in degrees. If omitted, inferred from caminfo after spiceinit.")
    parser.add_argument("--center-lon", type=float, default=None, help="Optional map center longitude in degrees positive east. If omitted, inferred from caminfo after spiceinit.")
    parser.add_argument("--tile-lat", type=float, default=None, help="Optional target latitude for deterministic fixed-tile cam2map output.")
    parser.add_argument("--tile-lon", type=float, default=None, help="Optional target longitude for deterministic fixed-tile cam2map output.")
    parser.add_argument("--tile-size-km", type=float, default=None, help="Optional deterministic tile size in km. Requires --tile-lat and --tile-lon.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for final GeoTIFF. Defaults to the standard processed location.")
    parser.add_argument("--img-url", default=None, help="Optional direct PDS IMG URL from ODE/PDS search; skips collection path rediscovery.")
    parser.add_argument("--pixel-resolution", type=float, default=0.5, help="Output map pixel resolution in meters/pixel")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete WSL temp workspace after processing")
    parser.add_argument("--max-volume", type=int, default=80, help="Maximum LROLRC volume number to search")
    parser.add_argument("--skip-if-exists", action="store_true", help="Return successfully without processing when the output GeoTIFF already exists")
    parser.add_argument("--no-isis-cache", action="store_true", help="Disable reuse of cached calibrated ISIS cubes")
    parser.add_argument("--refresh-isis-cache", action="store_true", help="Rebuild the cached calibrated ISIS cube for this product")
    parser.add_argument(
        "--shape",
        choices=["system", "ellipsoid", "user"],
        default=None,
        help="Optional ISIS spiceinit shape mode. Omitted preserves legacy behavior; user enables a supplied DEM/shape cube.",
    )
    parser.add_argument("--shape-model", default=None, help="WSL path to a DEM/shape cube for --shape user")
    parser.add_argument("--spksmithed", action="store_true", help="Use spiceinit spksmithed=true, as recommended by the LROC NAC processing guide when available")
    parser.add_argument("--require-smithed", action="store_true", help="Require smithed SPK kernels by adding spkrecon=false")
    parser.add_argument("--echo-correct", action="store_true", help="Run lronacecho after lronaccal before cam2map")
    parser.add_argument("--forwardpatch", action="store_true", help="Run cam2map with warpalgorithm=forwardpatch")
    parser.add_argument("--patch-size", type=int, default=100, help="cam2map forwardpatch patchsize")
    parser.add_argument(
        "--guide-pipeline",
        action="store_true",
        help="Shortcut for guide-style processing: --spksmithed --echo-correct --forwardpatch --patch-size 100",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the normalized product id and output plan without downloading or running WSL/ISIS")
    parser.add_argument("--verbose", action="store_true", help="Show verbose discovery logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    try:
        if args.guide_pipeline:
            args.spksmithed = True
            args.echo_correct = True
            args.forwardpatch = True
            args.patch_size = 100
        tile_map = None
        if args.tile_lat is not None or args.tile_lon is not None or args.tile_size_km is not None:
            if args.tile_lat is None or args.tile_lon is None or args.tile_size_km is None:
                raise ValueError("--tile-lat, --tile-lon, and --tile-size-km must be provided together.")
            tile_map = tile_map_bounds_from_latlon(args.tile_lat, args.tile_lon, args.tile_size_km)
        process_product(
            args.product_id,
            center_lat=args.center_lat,
            center_lon=args.center_lon,
            pixel_resolution=args.pixel_resolution,
            tile_map=tile_map,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            img_url=args.img_url,
            keep_temp=args.keep_temp,
            max_volume=args.max_volume,
            skip_if_exists=args.skip_if_exists,
            dry_run=args.dry_run,
            use_isis_cache=not args.no_isis_cache,
            refresh_isis_cache=args.refresh_isis_cache,
            shape=args.shape,
            shape_model=args.shape_model,
            spksmithed=args.spksmithed,
            require_smithed=args.require_smithed,
            echo_correct=args.echo_correct,
            forwardpatch=args.forwardpatch,
            patch_size=args.patch_size,
        )
    except subprocess.CalledProcessError as exc:
        LOGGER.error("\n[ERROR] Command failed with exit code %s", exc.returncode)
        return exc.returncode or 1
    except Exception as exc:
        LOGGER.error("\n[ERROR] %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
