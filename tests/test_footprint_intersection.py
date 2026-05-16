import geopandas as gpd
from shapely.geometry import Polygon

import pandas as pd

from lunar_tile_pipeline.lroc import (
    find_lroc_nac_edr_for_tile,
    parse_lroc_product_id,
    select_top_lroc_nac_for_tile,
)
from lunar_tile_pipeline.tiling import get_tile_for_latlon


def test_product_id_parsing():
    assert parse_lroc_product_id("M123456789LE") == {
        "product_id": "M123456789LE",
        "camera": "NAC-L",
        "processing_level": "EDR",
    }
    assert parse_lroc_product_id("M123456789RE")["camera"] == "NAC-R"
    assert parse_lroc_product_id("M123456789LC")["processing_level"] == "CDR"


def test_footprint_containing_tile_center_returns_product():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    gdf = gpd.GeoDataFrame(
        [{"PRODUCT_ID": "M123456789LE", "geometry": Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])}],
        geometry="geometry",
    )
    out = find_lroc_nac_edr_for_tile(tile, gdf, source_footprint_file="synthetic")
    assert out["product_id"].tolist() == ["M123456789LE"]
    assert out["contains_center"].tolist() == [True]
    assert out["full_tile_candidate"].tolist() == [True]


def test_footprint_overlapping_tile_edge_returns_product():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    gdf = gpd.GeoDataFrame(
        [{"PRODUCT_ID": "M123456789RE", "geometry": Polygon([(0.04, -0.1), (0.2, -0.1), (0.2, 0.1), (0.04, 0.1)])}],
        geometry="geometry",
    )
    out = find_lroc_nac_edr_for_tile(tile, gdf, source_footprint_file="synthetic")
    assert out["product_id"].tolist() == ["M123456789RE"]
    assert out["intersects_tile"].tolist() == [True]


def test_footprint_outside_tile_returns_no_product():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    gdf = gpd.GeoDataFrame(
        [{"PRODUCT_ID": "M123456789RE", "geometry": Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])}],
        geometry="geometry",
    )
    out = find_lroc_nac_edr_for_tile(tile, gdf, source_footprint_file="synthetic")
    assert out.empty


def test_cdr_source_id_is_converted_to_edr_for_search():
    tile = get_tile_for_latlon(0, 0, tile_size_km=10)
    gdf = gpd.GeoDataFrame(
        [{"PRODUCT_ID": "M123456789RC", "geometry": Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])}],
        geometry="geometry",
    )
    out = find_lroc_nac_edr_for_tile(tile, gdf, source_footprint_file="synthetic")
    assert out["product_id"].tolist() == ["M123456789RE"]


def test_tile_selection_prefers_full_tile_over_sharp_partial():
    products = pd.DataFrame(
        [
            {
                "product_id": "M000000001RE",
                "tile_coverage_fraction": 0.25,
                "resolution_m_per_pixel": 0.4,
                "incidence_angle": 30,
            },
            {
                "product_id": "M000000002RE",
                "tile_coverage_fraction": 0.98,
                "resolution_m_per_pixel": 1.0,
                "incidence_angle": 35,
            },
            {
                "product_id": "M000000003RE",
                "tile_coverage_fraction": 0.97,
                "resolution_m_per_pixel": 1.1,
                "incidence_angle": 65,
            },
        ]
    )
    selected = select_top_lroc_nac_for_tile(products, max_products=2)
    assert selected["product_id"].tolist() == ["M000000002RE", "M000000003RE"]
    assert selected["full_tile_candidate"].tolist() == [True, True]


def test_tile_selection_keeps_incidence_diversity_with_similar_full_tiles():
    products = pd.DataFrame(
        [
            {
                "product_id": "M000000001RE",
                "tile_coverage_fraction": 1.0,
                "resolution_m_per_pixel": 0.5,
                "incidence_angle": 30,
            },
            {
                "product_id": "M000000002RE",
                "tile_coverage_fraction": 0.99,
                "resolution_m_per_pixel": 0.55,
                "incidence_angle": 32,
            },
            {
                "product_id": "M000000003RE",
                "tile_coverage_fraction": 0.98,
                "resolution_m_per_pixel": 0.6,
                "incidence_angle": 70,
            },
        ]
    )
    selected = select_top_lroc_nac_for_tile(products, max_products=2)
    assert selected["product_id"].tolist() == ["M000000001RE", "M000000003RE"]
