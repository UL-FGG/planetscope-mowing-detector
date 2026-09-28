"""Build PlanetCube.nc from an existing Planet/Sentinel Hub BYOC collection."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sentinelhub import (
    BBox,
    CRS,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)

from .aoi import create_square_aoi
from .config import WorkflowConfig
from .io import save_netcdf
from .utils import get_logger

LOGGER = get_logger(__name__)


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["blue", "green", "red", "rededge", "nir", "clear", "cloud", "shadow", "snow", "dataMask"]
    }],
    output: {
      bands: 10,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.blue,
    sample.green,
    sample.red,
    sample.rededge,
    sample.nir,
    sample.clear,
    sample.cloud,
    sample.shadow,
    sample.snow,
    sample.dataMask
  ];
}
"""


def sentinelhub_config(client_id: str | None = None, client_secret: str | None = None) -> SHConfig:
    """Create a Sentinel Hub config, optionally overriding stored credentials."""
    config = SHConfig()
    if client_id:
        config.sh_client_id = client_id
    if client_secret:
        config.sh_client_secret = client_secret
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError("Sentinel Hub OAuth credentials are required.")
    return config


def request_collection_day(
    collection_id: str,
    bbox: BBox,
    size: tuple[int, int],
    date: pd.Timestamp,
    config: SHConfig,
) -> np.ndarray:
    """Request one day from an existing BYOC collection."""
    data_collection = DataCollection.define_byoc(collection_id)
    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=(date.strftime("%Y-%m-%d"), (date + timedelta(days=1)).strftime("%Y-%m-%d")),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    return request.get_data()[0]


def build_collection_cube(
    workflow: WorkflowConfig,
    collection_id: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    resolution: float = 3.0,
) -> Path:
    """Build PlanetCube.nc from an existing 8-band Surface Reflectance collection."""
    workflow.create_directories()
    aoi = create_square_aoi(
        workflow.center_lat,
        workflow.center_lon,
        workflow.aoi_size,
        workflow.aoi_dir / f"aoi_{workflow.aoi_size}m.geojson",
    )
    bbox = BBox(bbox=aoi["bbox"], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    config = sentinelhub_config(client_id, client_secret)

    dates = pd.date_range(workflow.start_date, workflow.end_date, freq="D")
    datasets: list[xr.Dataset] = []
    x = np.arange(size[0], dtype="float32")
    y = np.arange(size[1], dtype="float32")

    for date in dates:
        try:
            data = request_collection_day(collection_id, bbox, size, date, config)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", date.date(), exc)
            continue

        if data.ndim != 3 or data.shape[-1] != 10:
            LOGGER.warning("Skipping %s: unexpected shape %s", date.date(), data.shape)
            continue
        if np.nanmax(data[:, :, 9]) == 0:
            continue

        quality = (
            (data[:, :, 5] == 1)
            & (data[:, :, 6] == 0)
            & (data[:, :, 7] == 0)
            & (data[:, :, 8] == 0)
            & (data[:, :, 9] == 1)
        ).astype("uint8")

        dataset = xr.Dataset(
            {
                "blue": (("y", "x"), data[:, :, 0].astype("float32")),
                "green": (("y", "x"), data[:, :, 1].astype("float32")),
                "red": (("y", "x"), data[:, :, 2].astype("float32")),
                "rededge": (("y", "x"), data[:, :, 3].astype("float32")),
                "nir": (("y", "x"), data[:, :, 4].astype("float32")),
                "quality": (("y", "x"), quality),
            },
            coords={"y": y, "x": x},
            attrs={
                "collection_id": collection_id,
                "crs": "EPSG:4326",
                "resolution": resolution,
                "bbox": aoi["bbox"],
                "source": "existing_byoc_collection",
            },
        ).expand_dims(time=[pd.Timestamp(date)])
        dataset = dataset.assign_coords(scene_id=("time", [f"{collection_id}_{date.strftime('%Y%m%d')}"]))
        datasets.append(dataset)
        LOGGER.info("Added %s", date.date())

    if not datasets:
        raise RuntimeError("No valid observations were returned for the selected AOI and date range.")

    cube = xr.concat(datasets, dim="time").sortby("time")
    cube.attrs["description"] = "PlanetScope Surface Reflectance cube from existing BYOC collection."
    save_netcdf(cube, workflow.planet_cube_path)
    LOGGER.info("Saved Planet cube to %s", workflow.planet_cube_path)
    return workflow.planet_cube_path
