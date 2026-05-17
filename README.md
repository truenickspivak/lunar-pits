# Lunar Pits

Tools for gathering, processing, tiling, and organizing LROC NAC imagery and
coarse lunar context data for skylight/lava-tube entrance detection.

The repo is currently centered around a coordinate-first workflow: give it a
lunar latitude/longitude, it finds NAC observations that actually cover the
deterministic tile for that coordinate, processes the best observations through
ISIS, and writes one inspectable training unit under `data/locations/`.

## Current Pipeline

The main training unit is a location folder:

```text
data/locations/<location_id>/
  tile.json
  metadata.json
  manifest.csv
  manifest.parquet
  annotations.csv
  nac/
    products.csv
    products.parquet
    images/
      <tile_id>_<PRODUCT_ID>_ml.tif
      <tile_id>_<PRODUCT_ID>_ml_preview.png
      <tile_id>_<PRODUCT_ID>.json
    mapped/
      <PRODUCT_ID>.map.tif
    masks/
      README.md
```

`tile.json` is the compact source of truth for CNN training. It records the
target coordinate, deterministic tile metadata, selected NAC observations,
label metadata, mask status, and coarse context from GRAIL, LOLA, and Diviner
when those sources are available.

## One Coordinate

Run a single coordinate gather:

```powershell
cd C:\Users\nicks\Desktop\lunar-pits
conda run -n lunar python scripts\gather_location_context.py --lat 8.3355 --lon 33.222 --radius-km 5 --max-nac 5 --process-nac --download-context
```

Useful options:

```text
--dry-run              Search/write metadata without running ISIS.
--max-nac 5            Number of ranked NAC observations to process.
--tile-size-km 0.256   Deterministic tile size. Default is 256 m.
--pixel-resolution 0.5 Output pixel size for NAC ML tiles.
--force                Rebuild existing location products.
```

The gatherer uses NAC footprint polygons. Image center coordinates are treated
as metadata only, because NAC strips are long and narrow and the center is not
proof that a target coordinate is covered.

## Full Dataset Runner

For the full skylight dataset, use the root-level double-click runner:

```text
RUN_SKYLIGHT_DATASET.bat
```

It activates the `lunar` conda environment, starts a timestamped log under
`logs/`, and calls:

```powershell
conda run -n lunar python scripts\build_skylight_dataset.py
```

The runner creates and updates:

```text
data/dataset_queue.csv
```

That CSV is the progress board. It includes every coordinate to do and already
done, with columns such as:

```text
queue_index, split_group, label, label_source, name, lat, lon_180, lon_360,
location_id, status, started_at, completed_at, num_nac_requested,
num_nac_completed, location_path, tile_json, error
```

Rows are processed in this order:

1. `positive`
2. `negative`
3. `undetermined`

Statuses are:

```text
pending, running, complete, failed, skipped
```

If the run stops, rerun the same `.bat`. Completed rows are skipped and the
runner resumes from unfinished rows.

Dry-run only builds/checks the queue:

```powershell
.\RUN_SKYLIGHT_DATASET.bat --dry-run --force-queue-rebuild
```

Current default queue balance is:

```text
positive:      all official LROC Lunar Pit Locations rows
negative:      3x positives
undetermined:  1x positives
```

In this environment the official LROC pit CSV currently loads as 278 valid
positive rows, so the default dry-run queue is:

```text
278 positive
834 negative
278 undetermined
1390 total coordinate units
```

The code does not hardcode 278 or 283; it uses whatever valid rows are in the
official cached LROC catalog.

## Labels And Masks

Positive LROC catalog rows are labeled:

```text
positive_skylight_candidate
```

The original catalog metadata is preserved where available, including name,
terrain, type, overhang notes, useful image IDs, incidence, emission,
resolution, and description fields.

Automatically generated negatives are labeled:

```text
negative_auto
```

Automatically generated ambiguous/hard examples are labeled:

```text
undetermined_auto
```

Positive rows still need manual instance masks before final supervised Mask
R-CNN-style training. Negative and undetermined rows can be complete without
masks for initial collection. Train/validation/test splits should be made by
coordinate or tile, not by individual NAC image, to avoid leakage.

## LROC NAC Processing

Single-product processing is still available:

```powershell
python scripts\process_lroc_product.py M107689129RE --skip-if-exists
```

The processor normalizes product IDs, finds the PDS IMG, downloads it to WSL
temporary storage, runs the ISIS conversion/calibration/map-projection path,
copies the final GeoTIFF back into the project, and deletes temporary WSL files
after the final GeoTIFF exists.

The working ISIS invocation is based on:

```bash
~/miniforge3/bin/conda run -n isis9.0.0 <isis-command>
```

The location gatherer uses this processor as the execution engine. For ML
tiles, it maps each selected NAC observation once, then crops deterministic
post-map tiles rather than running `cam2map` separately for every tiny tile.

## Deterministic Tiles

The first deterministic grid uses a spherical lunar equirectangular
approximation:

```text
Moon radius = 1737400 m
Longitude convention = positive east
Global origin = lon 0, lat 0
```

For a fixed tile size, the same coordinate always maps to the same tile ID and
pixel offset. This is the key invariant for comparing the same location across
multiple NAC observations with different lighting.

The older `lunar-tiles` CLI still exists for lower-level tile experiments:

```powershell
lunar-tiles process-point --lat 14.123 --lon 303.456 --tile-size-km 10 --footprints path\to\lroc_nac_edr_footprints.shp --out data\tiles
lunar-tiles process-region --min-lat 10 --max-lat 20 --min-lon 300 --max-lon 320 --tile-size-km 10 --footprints path\to\footprints.shp --out data\tiles
lunar-tiles list-global --tile-size-km 10 --out global_10km_tiles.csv
```

For current pit/skylight work, prefer `scripts\gather_location_context.py` and
`RUN_SKYLIGHT_DATASET.bat`.

## Context Data

The context source config is:

```text
config/context_sources.yaml
```

The samplers currently support:

```text
topology: LOLA
gravity:  GRAIL/Bouguer-style sources
ir:       Diviner thermal anomaly/context sources
```

If local or cached sources are missing, `tile.json` records an explicit
`available: false` payload with the reason and TODO. When configured or cached,
the samplers store simple statistics such as min, max, mean, median, standard
deviation, valid fraction, source file, and coarse native-resolution notes.

## QA And Utility Scripts

Useful scripts:

```powershell
python scripts\qa_lroc_tif.py data\processed\lroc_tif\M114328462RE.map.tif
python scripts\tile_lroc_tif.py data\processed\lroc_tif\M114328462RE.map.tif --tile-size-m 512 --stride-m 256
python scripts\inspect_tiles.py
python scripts\verify_ml_tile_alignment.py
```

The QA scripts are for inspection and debugging. Production CNN data should use
the fixed location-folder outputs and the scaling policy recorded in each tile
sidecar/metadata file.

## Tests

Run tests from the repo root with `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH='src'
conda run -n lunar python -m unittest discover -s tests
```

Current tests cover product ID/path utilities, footprint matching, deterministic
tiling, ML tile geometry/scaling checks, location utilities, and the resumable
skylight dataset queue.

## Current Limitations

- The deterministic grid is currently a spherical Moon approximation. It is not
  an Earth CRS.
- Polar regions may need specialized projection handling later.
- Negative and undetermined coordinate generation is deterministic but still
  automatic; review is recommended before final training conclusions.
- Positive rows need manual masks before supervised instance-segmentation
  training.
- Coarse LOLA/GRAIL/Diviner sources may represent much larger spatial support
  than a 256 m NAC tile. The metadata records that mismatch rather than hiding
  it.
