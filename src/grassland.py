"""Grassland mask preparation utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def clip_and_dissolve_grasslands(
    grassland_gpkg: Path,
    aoi_geojson: Path,
    output_gpkg: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Clip a national grassland mask to an AOI and dissolve it."""
    if output_gpkg.exists() and not overwrite:
        return output_gpkg
    if not grassland_gpkg.exists():
        raise FileNotFoundError(f"Grassland mask not found: {grassland_gpkg}")
    if not aoi_geojson.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_geojson}")

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    aoi = gpd.read_file(aoi_geojson)
    grasslands = gpd.read_file(grassland_gpkg, mask=aoi).to_crs(aoi.crs)
    clipped = gpd.clip(grasslands, aoi)
    if not clipped.empty:
        clipped = clipped.dissolve().reset_index(drop=True)
    clipped.to_file(output_gpkg, driver="GPKG")
    return output_gpkg
