"""User settings for the PlanetScope mowing detector package."""

from pathlib import Path

# Area definition
AREA_ID = "area_01_2025"
CENTER_LAT = 46.28735
CENTER_LON = 14.00218
AOI_SIZE = 1000

# Season to process
START_DATE = "2025-03-01"
END_DATE = "2025-10-31"

# Existing Planet/Sentinel Hub BYOC collection with 8-band PlanetScope SR data.
COLLECTION_ID = "8523c8f2-e983-4a9c-8e69-88365d946cbf"

# Output folder inside this package.
DATA_ROOT = Path("data")

# Processing options
RESOLUTION = 3.0
FORCE_REBUILD = False
REBUILD_PLANET_CUBE = False
REBUILD_INDICES_CUBE = False
REBUILD_DETECTION = False

# National grassland mask. The build script clips and dissolves it to the AOI
# and stores the result as data/<AREA_ID>/masks/grasslands_clipped.gpkg.
USE_GRASSLAND_MASK = True
GRASSLAND_SOURCE = Path("data/grassland/travniki_RABA_GERK_20260831.gpkg")
OVERWRITE_CLIPPED_GRASSLAND_MASK = False

# Segment filtering
MIN_COMPONENT_PIXELS = 10
