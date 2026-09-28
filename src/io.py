"""Input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file, returning an empty dict if it is missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def open_dataset(path: Path) -> xr.Dataset:
    """Open an existing NetCDF dataset with a helpful error."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return xr.open_dataset(path)


def save_netcdf(dataset: xr.Dataset, path: Path, compression_level: int = 4) -> None:
    """Save a compressed NetCDF4 dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        variable: {"zlib": True, "complevel": compression_level}
        for variable in dataset.data_vars
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
