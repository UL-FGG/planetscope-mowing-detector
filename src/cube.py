"""Build xarray cubes from PlanetScope imagery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import xy

from .io import save_netcdf
from .utils import get_logger

LOGGER = get_logger(__name__)

BAND_MAP = {"blue": 2, "green": 4, "red": 6, "rededge": 7, "nir": 8}


def find_reflectance_files(raw_dir: Path) -> list[Path]:
    """Find PlanetScope 8-band Surface Reflectance GeoTIFFs."""
    return sorted(
        path for path in raw_dir.rglob("*.tif")
        if "udm" not in path.name.lower() and ("analytic" in path.name.lower() or "sr" in path.name.lower())
    )


def find_quality_file(scene_path: Path) -> Path | None:
    """Find a nearby UDM2 quality mask for a scene."""
    siblings = list(scene_path.parent.glob("*udm2*.tif")) + list(scene_path.parent.glob("*UDM2*.tif"))
    return siblings[0] if siblings else None


def _coordinates(src: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([xy(src.transform, 0, col, offset="center")[0] for col in range(src.width)], dtype="float64")
    ys = np.array([xy(src.transform, row, 0, offset="center")[1] for row in range(src.height)], dtype="float64")
    return xs, ys


def _acquisition_date(path: Path, features: list[dict[str, Any]]) -> pd.Timestamp:
    for feature in features:
        if feature["id"] in path.name:
            return pd.to_datetime(feature["properties"]["acquired"])
    return pd.NaT


def build_planet_cube(raw_dir: Path, output_path: Path, features: list[dict[str, Any]]) -> Path:
    """Create PlanetCube.nc with blue, green, red, rededge, nir, and quality variables."""
    files = find_reflectance_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No PlanetScope reflectance GeoTIFFs found in {raw_dir}")

    datasets: list[xr.Dataset] = []
    for path in files:
        with rasterio.open(path) as src:
            xs, ys = _coordinates(src)
            data_vars = {}
            for name, band_index in BAND_MAP.items():
                data = src.read(band_index).astype("float32")
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                data_vars[name] = (("y", "x"), data)

            quality_path = find_quality_file(path)
            if quality_path:
                with rasterio.open(quality_path) as qsrc:
                    quality = qsrc.read(1).astype("uint16")
            else:
                quality = np.ones((src.height, src.width), dtype="uint16")

            scene_id = next((feature["id"] for feature in features if feature["id"] in path.name), path.stem)
            acquisition = _acquisition_date(path, features)
            dataset = xr.Dataset(
                {**data_vars, "quality": (("y", "x"), quality)},
                coords={"y": ys, "x": xs},
                attrs={
                    "crs": src.crs.to_string() if src.crs else "",
                    "pixel_size": (src.transform.a, abs(src.transform.e)),
                    "transform": tuple(src.transform),
                    "source": str(path),
                },
            ).expand_dims(time=[acquisition])
            dataset = dataset.assign_coords(scene_id=("time", [scene_id]))
            datasets.append(dataset)

    cube = xr.concat(datasets, dim="time").sortby("time")
    cube.attrs["description"] = "PlanetScope Surface Reflectance pixel time-series cube."
    save_netcdf(cube, output_path)
    LOGGER.info("Saved Planet cube to %s", output_path)
    return output_path
