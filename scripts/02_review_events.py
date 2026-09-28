"""Launch the mowing event reviewer."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import settings
from src.config import WorkflowConfig
from src.product import review_events

workflow = WorkflowConfig(
    center_lat=settings.CENTER_LAT,
    center_lon=settings.CENTER_LON,
    aoi_size=settings.AOI_SIZE,
    start_date=settings.START_DATE,
    end_date=settings.END_DATE,
    output_folder=PROJECT_ROOT / settings.DATA_ROOT,
    area_id=settings.AREA_ID,
)

review_events(
    indices_cube_path=workflow.indices_cube_path,
    planet_cube_path=workflow.planet_cube_path,
    detection_dir=workflow.outputs_dir / "product_detection",
    catalog_csv=workflow.outputs_dir / "events" / "event_catalog.csv",
    output_csv=workflow.outputs_dir / "events" / "event_reviews.csv",
)
