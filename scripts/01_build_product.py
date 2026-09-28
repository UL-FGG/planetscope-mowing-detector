"""Build PlanetScope cube, index cube, mowing rasters and event catalog."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import settings
from src.aoi import create_square_aoi
from src.config import WorkflowConfig
from src.grassland import clip_and_dissolve_grasslands
from src.product import build_event_catalog, run_product
from src.timeseries_detection import PlanetTimeSeriesDetectionParameters


def _credential(name: str, prompt: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value
    return getpass(prompt)


workflow = WorkflowConfig(
    center_lat=settings.CENTER_LAT,
    center_lon=settings.CENTER_LON,
    aoi_size=settings.AOI_SIZE,
    start_date=settings.START_DATE,
    end_date=settings.END_DATE,
    output_folder=PROJECT_ROOT / settings.DATA_ROOT,
    area_id=settings.AREA_ID,
)

parameters = PlanetTimeSeriesDetectionParameters(
    min_ndvi_drop=0.10,
    min_ndre_drop=0.04,
    min_gndvi_drop=0.06,
    min_indices_with_drop=1,
    min_recovery_ndvi_gain=0.05,
    sgs_drop_threshold=0.093,
    sgs_growth_ratio=1.4,
    sgs_min_event_spacing_days=15,
    sgs_rapid_rebound_days=5,
    sgs_rapid_rebound_threshold=0.15,
    sgs_smoothing_window=3,
    final_min_neighbour_events=3,
    final_min_component_pixels=settings.MIN_COMPONENT_PIXELS,
    final_neighbour_date_tolerance_days=7,
    detection_start_month=4,
    detection_start_day=1,
    detection_end_month=10,
    detection_end_day=15,
)

sh_client_id = _credential("SH_CLIENT_ID", "Sentinel Hub client ID: ")
sh_client_secret = _credential("SH_CLIENT_SECRET", "Sentinel Hub client secret: ")

create_square_aoi(
    workflow.center_lat,
    workflow.center_lon,
    workflow.aoi_size,
    workflow.aoi_dir / f"aoi_{workflow.aoi_size}m.geojson",
)

grassland_mask = None
if settings.USE_GRASSLAND_MASK:
    grassland_mask = clip_and_dissolve_grasslands(
        PROJECT_ROOT / settings.GRASSLAND_SOURCE,
        workflow.aoi_dir / f"aoi_{workflow.aoi_size}m.geojson",
        workflow.masks_dir / "grasslands_clipped.gpkg",
        overwrite=settings.OVERWRITE_CLIPPED_GRASSLAND_MASK,
    )
    print(f"Using grassland mask: {grassland_mask}")

outputs = run_product(
    workflow,
    collection_id=settings.COLLECTION_ID,
    sentinelhub_client_id=sh_client_id,
    sentinelhub_client_secret=sh_client_secret,
    resolution=settings.RESOLUTION,
    grassland_mask=grassland_mask,
    parameters=parameters,
    force_rebuild=settings.FORCE_REBUILD,
    rebuild_planet_cube=settings.REBUILD_PLANET_CUBE,
    rebuild_indices_cube=settings.REBUILD_INDICES_CUBE,
    rebuild_detection=settings.REBUILD_DETECTION,
)

event_catalog = build_event_catalog(
    workflow.indices_cube_path,
    outputs["detection_dir"],
    workflow.outputs_dir / "events" / "event_catalog.csv",
    min_component_pixels=settings.MIN_COMPONENT_PIXELS,
)

print("Done.")
for name, path in outputs.items():
    print(f"{name}: {path}")
print(f"event_catalog: {event_catalog}")
