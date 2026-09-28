"""AOI creation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Point, box, mapping


def create_square_aoi(
    center_lat: float,
    center_lon: float,
    size_m: int,
    output_path: Path,
    utm_epsg: int = 32633,
) -> dict[str, Any]:
    """Create a square AOI in metres and save it as WGS84 GeoJSON."""
    if size_m <= 0:
        raise ValueError("size_m must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(utm_epsg)
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    center_x, center_y = transformer.transform(center_lon, center_lat)
    half_size = size_m / 2
    geometry_utm = box(center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)

    aoi_utm = gpd.GeoDataFrame({"name": [f"aoi_{size_m}m"]}, geometry=[geometry_utm], crs=utm)
    aoi_wgs84 = aoi_utm.to_crs(wgs84)
    aoi_wgs84.to_file(output_path, driver="GeoJSON")

    center = gpd.GeoDataFrame({"name": ["center"]}, geometry=[Point(center_lon, center_lat)], crs=wgs84)
    geometry = aoi_wgs84.geometry.iloc[0]
    return {
        "aoi": aoi_wgs84,
        "center": center,
        "geometry": mapping(geometry),
        "bbox": aoi_wgs84.total_bounds.tolist(),
        "wkt": geometry.wkt,
        "path": output_path,
    }
