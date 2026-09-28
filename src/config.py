"""Configuration models for the mowing workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowConfig:
    """User-editable workflow configuration."""

    center_lat: float
    center_lon: float
    aoi_size: int
    start_date: str
    end_date: str
    output_folder: Path
    area_id: str = "area_01"

    def __post_init__(self) -> None:
        if not -90 <= self.center_lat <= 90:
            raise ValueError("center_lat must be between -90 and 90.")
        if not -180 <= self.center_lon <= 180:
            raise ValueError("center_lon must be between -180 and 180.")
        if self.aoi_size <= 0:
            raise ValueError("aoi_size must be positive.")

    @property
    def root(self) -> Path:
        return Path(self.output_folder) / self.area_id

    @property
    def aoi_dir(self) -> Path:
        return self.root / "aoi"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def cube_dir(self) -> Path:
        return self.root / "cube"

    @property
    def indices_dir(self) -> Path:
        return self.root / "indices"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def masks_dir(self) -> Path:
        return self.root / "masks"

    @property
    def planet_cube_path(self) -> Path:
        return self.cube_dir / "PlanetCube.nc"

    @property
    def indices_cube_path(self) -> Path:
        return self.indices_dir / "IndicesCube.nc"

    def create_directories(self) -> None:
        """Create the standard data folders."""
        for folder in [
            self.aoi_dir,
            self.raw_dir,
            self.cube_dir,
            self.indices_dir,
            self.outputs_dir,
            self.figures_dir,
            self.masks_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)
