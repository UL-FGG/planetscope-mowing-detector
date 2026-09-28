"""Vegetation index calculations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import xarray as xr

from .io import open_dataset, save_netcdf

IndexFunction = Callable[[xr.Dataset], xr.DataArray]


def normalized_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """Calculate a normalized difference index."""
    return (a - b) / (a + b)


def ndvi(cube: xr.Dataset) -> xr.DataArray:
    """Normalized Difference Vegetation Index."""
    return normalized_difference(cube["nir"], cube["red"])


def ndre(cube: xr.Dataset) -> xr.DataArray:
    """Normalized Difference Red Edge Index."""
    return normalized_difference(cube["nir"], cube["rededge"])


def gndvi(cube: xr.Dataset) -> xr.DataArray:
    """Green Normalized Difference Vegetation Index."""
    return normalized_difference(cube["nir"], cube["green"])


INDEX_REGISTRY: dict[str, IndexFunction] = {
    "ndvi": ndvi,
    "ndre": ndre,
    "gndvi": gndvi,
}


def build_indices(planet_cube_path: Path, output_path: Path, indices: list[str] | None = None) -> Path:
    """Build an index cube from PlanetCube.nc.

    If the input cube contains a ``quality`` variable, all indices are masked so
    that only pixels with ``quality == 1`` remain valid. Cloud, shadow, snow and
    NoData observations therefore become NaN in the output index cube.
    """
    selected = indices or list(INDEX_REGISTRY)
    cube = open_dataset(planet_cube_path)
    missing = [name for name in selected if name not in INDEX_REGISTRY]
    if missing:
        raise ValueError(f"Unknown indices: {missing}")
    quality_mask = cube["quality"] == 1 if "quality" in cube else None
    data_vars = {}
    for name in selected:
        values = INDEX_REGISTRY[name](cube)
        if quality_mask is not None:
            values = values.where(quality_mask)
        data_vars[name] = values

    if "quality" in cube:
        data_vars["quality"] = cube["quality"]

    index_cube = xr.Dataset(data_vars)
    index_cube.attrs.update(cube.attrs)
    index_cube.attrs["description"] = "Vegetation index pixel time-series cube masked by quality == 1."
    save_netcdf(index_cube, output_path)
    return output_path
