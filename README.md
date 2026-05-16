# Lunar Pits

Finding all the lunar lava tubes on the Moon.

## Deterministic Lunar Tile Pipeline

This repo includes a first-pass `lunar_tile_pipeline` package for building
coordinate-stable lunar data tiles. The key rule is that processing is based on
fixed tile indices, not ad-hoc latitude/longitude windows. The same coordinate
always maps to the same tile ID for a given tile size.

The first version uses a spherical lunar equirectangular approximation with
`R = 1737400 m`. This is not an Earth CRS and should not be treated as
EPSG:4326. It is a deterministic global indexing frame that can be swapped for
a more precise lunar projection later.

Example:

```powershell
lunar-tiles process-point --lat 14.123 --lon 303.456 --tile-size-km 10 --footprints path\to\lroc_nac_edr_footprints.shp --out data\tiles
```

This writes:

```text
data/tiles/moon_10km_x+000XXX_y+000YYY/
  tile_metadata.json
  lroc_nac_edr_products.csv
  lroc_nac_edr_products.json
  nac_crops/
  grail/
  diviner/
  lola/
  masks/
  labels.json
```

Other commands:

```powershell
lunar-tiles process-region --min-lat 10 --max-lat 20 --min-lon 300 --max-lon 320 --tile-size-km 10 --footprints path\to\footprints.shp --out data\tiles
lunar-tiles list-global --tile-size-km 10 --out global_10km_tiles.csv
lunar-tiles inspect-tile --tile-id moon_10km_x+000123_y-000045
```

LROC NAC matching requires footprint polygons. Image center coordinates are not
sufficient because NAC strips are long and narrow; a coordinate can be inside
the image even when far from the image center.

For high-resolution NAC pit work, smaller fixed grids are often more useful
than 10 km grids. NAC images are long, narrow strips, so many products will only
partly intersect a 10 km square. Use `--tile-size-km 1` or `--tile-size-km 2`
when the goal is a repeatable high-resolution ML chip around a feature; the
tile ID is still deterministic from the global 0 latitude / 0 longitude grid.
The NAC ranking favors full-tile coverage first, then comparable best
resolution, then varied incidence angles. If no product fully covers a larger
tile, it falls back to the widest usable coverage with sane viewing geometry.

Current limitations:

- 10 km, 5 km, 2 km, and 1 km grids are deterministic, but the first projection is a
  spherical Moon approximation.
- Polar regions may need specialized projection handling later.
- NAC crop/reprojection is currently a planned-output placeholder.
- GRAIL, Diviner, and LOLA modules expose structured placeholders until local
  dataset rasters are configured and resampling is implemented.
