"""Time-series based PlanetScope mowing detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy.ndimage import label as connected_components
from scipy.signal import find_peaks
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from shapely.ops import transform as transform_geometry

from .io import open_dataset


@dataclass(frozen=True)
class PlanetTimeSeriesDetectionParameters:
    """Parameters for transparent mowing detection on candidate polygons."""

    minimum_valid_fraction: float = 0.5
    min_ndvi_drop: float = 0.14
    min_ndre_drop: float = 0.05
    min_gndvi_drop: float = 0.08
    min_indices_with_drop: int = 2
    max_quick_recovery_ratio: float = 0.8
    max_next_ndvi_tolerance: float = 0.05
    min_pre_event_ndvi: float = 0.45
    min_recovery_days: int = 10
    max_recovery_days: int = 40
    min_recovery_ratio: float = 0.45
    min_recovery_ndvi_gain: float = 0.05
    min_area_pixels: int = 25
    deduplicate_days: int = 20
    sgs_drop_threshold: float = 0.093
    sgs_growth_ratio: float = 1.4
    sgs_min_event_spacing_days: int = 15
    sgs_rapid_rebound_days: int = 5
    sgs_rapid_rebound_threshold: float = 0.15
    sgs_smoothing_window: int = 3
    final_min_neighbour_events: int = 3
    final_min_component_pixels: int = 5
    final_neighbour_date_tolerance_days: int = 7
    detection_start_month: int = 4
    detection_start_day: int = 1
    detection_end_month: int = 10
    detection_end_day: int = 15


@dataclass(frozen=True)
class PlanetRasterDetectionOutputs:
    """Paths written by pixel-wise raster detection."""

    output_dir: Path
    rasters: dict[str, Path]
    events_nc: Path


def _detect_sgs_events(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    parameters: PlanetTimeSeriesDetectionParameters,
) -> list[tuple[int, float]]:
    """Sentinel-compatible peak/drop/recovery detector for one pixel."""
    return [(event_index, drop) for event_index, _, _, drop in _detect_sgs_event_details(dates, values, parameters)]


def _detect_sgs_event_details(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    parameters: PlanetTimeSeriesDetectionParameters,
) -> list[tuple[int, int, int, float]]:
    """Return valley, peak, recovery and drop for Sentinel-style SGS events."""
    values = np.asarray(values, dtype="float32")
    valid = np.isfinite(values)
    if valid.sum() < 3:
        return []
    f = values[valid]
    t = dates[valid]
    if parameters.sgs_smoothing_window > 1 and len(f) >= parameters.sgs_smoothing_window:
        f = (
            pd.Series(f)
            .rolling(parameters.sgs_smoothing_window, center=True, min_periods=1)
            .median()
            .to_numpy(dtype="float32")
        )
    peaks = find_peaks(f)[0]
    valleys = find_peaks(-f)[0]
    if not len(peaks) or not len(valleys):
        return []
    if valleys[0] < peaks[0]:
        peaks = np.insert(peaks, 0, 0)
    if valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(f) - 1)
    pairs = []
    valley_pos = 0
    for peak in peaks:
        while valley_pos < len(valleys) and valleys[valley_pos] <= peak:
            valley_pos += 1
        if valley_pos >= len(valleys):
            break
        pairs.append([peak, valleys[valley_pos]])
        valley_pos += 1
    if not pairs:
        return []
    merged = [np.asarray(pairs[0], dtype=int)]
    for peak, valley in pairs[1:]:
        old_peak, old_valley = merged[-1]
        if (
            f[peak] - f[old_valley] < parameters.sgs_drop_threshold * parameters.sgs_growth_ratio
            and f[valley] < f[old_valley]
            and f[peak] < f[old_peak]
        ):
            merged[-1][1] = valley
        else:
            merged.append(np.array([peak, valley]))
    events: list[tuple[int, int, int, float]] = []
    for peak, valley in merged:
        drop = float(f[peak] - f[valley])
        if drop <= parameters.sgs_drop_threshold:
            continue
        rebound_days = (t[min(len(t) - 1, valley + 1)] - t[valley]).days if valley + 1 < len(t) else 999
        rebound = float(f[min(len(f) - 1, valley + 1)] - f[valley]) if valley + 1 < len(f) else 0.0
        if rebound_days <= parameters.sgs_rapid_rebound_days and rebound > parameters.sgs_rapid_rebound_threshold:
            continue
        recovery = None
        for index in range(valley, len(f)):
            if f[index] - f[valley] > parameters.sgs_drop_threshold * parameters.sgs_growth_ratio:
                recovery = index
                break
        if recovery is None:
            continue
        event_day = t[valley]
        event_index = int(np.where(dates == event_day)[0][0])
        peak_index = int(np.where(dates == t[peak])[0][0])
        recovery_index = int(np.where(dates == t[recovery])[0][0])
        if events and (event_day - dates[events[-1][0]]).days <= parameters.sgs_min_event_spacing_days:
            if drop > events[-1][3]:
                events[-1] = (event_index, peak_index, recovery_index, drop)
        else:
            events.append((event_index, peak_index, recovery_index, drop))
    return events


def _pixel_polygon(row: pd.Series) -> Polygon:
    coords = json.loads(row["geometry_pixels"])
    return Polygon(coords)


def _polygon_mask(row: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    return rasterize(
        [(_pixel_polygon(row), 1)],
        out_shape=shape,
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _series_for_polygon(cube: xr.Dataset, row: pd.Series, index_name: str) -> tuple[np.ndarray, np.ndarray]:
    values = cube[index_name].values
    mask = _polygon_mask(row, values.shape[1:])
    segment = values[:, mask]
    finite = np.isfinite(segment)
    counts = finite.sum(axis=1)
    sums = np.nansum(segment, axis=1)
    mean = np.divide(sums, counts, out=np.full(len(counts), np.nan, dtype=float), where=counts > 0)
    valid_fraction = counts / segment.shape[1] if segment.shape[1] else np.zeros(len(counts), dtype=float)
    return mean, valid_fraction


def _next_valid_index(values: np.ndarray, valid: np.ndarray, start: int, minimum: float) -> int | None:
    for idx in range(start, len(values)):
        if valid[idx] >= minimum and np.isfinite(values[idx]):
            return idx
    return None


def _pixel_to_geo_polygon(cube: xr.Dataset, polygon: Polygon) -> Polygon | None:
    if "bbox" not in cube.attrs:
        return None
    minx, miny, maxx, maxy = [float(value) for value in cube.attrs["bbox"]]
    transform = from_bounds(minx, miny, maxx, maxy, int(cube.sizes["x"]), int(cube.sizes["y"]))

    def convert(x: float, y: float, z: float | None = None):
        lon, lat = transform * (x, y)
        return lon, lat

    return transform_geometry(convert, polygon)


def _deduplicate_events(events: pd.DataFrame, days: int) -> pd.DataFrame:
    if events.empty:
        return events
    result = []
    events = events.sort_values(["segment_key", "event_date", "score"], ascending=[True, True, False])
    for _, group in events.groupby("segment_key", sort=False):
        kept: list[pd.Series] = []
        for _, row in group.iterrows():
            event_date = pd.Timestamp(row["event_date"])
            overlaps = [
                abs((event_date - pd.Timestamp(existing["event_date"])).days) <= days
                for existing in kept
            ]
            if overlaps and any(overlaps):
                best_idx = next(idx for idx, value in enumerate(overlaps) if value)
                if float(row["score"]) > float(kept[best_idx]["score"]):
                    kept[best_idx] = row
            else:
                kept.append(row)
        result.extend(kept)
    return pd.DataFrame(result).sort_values(["event_date", "score"], ascending=[True, False]).reset_index(drop=True)


def _cube_transform(cube: xr.Dataset):
    if "bbox" not in cube.attrs:
        raise ValueError("Cube is missing bbox metadata required for GeoTIFF export.")
    minx, miny, maxx, maxy = [float(value) for value in cube.attrs["bbox"]]
    return from_bounds(minx, miny, maxx, maxy, int(cube.sizes["x"]), int(cube.sizes["y"]))


def _write_raster(path: Path, array: np.ndarray, cube: xr.Dataset, dtype: str, nodata: float | int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(cube.sizes["y"]),
        width=int(cube.sizes["x"]),
        count=1,
        dtype=dtype,
        crs=cube.attrs.get("crs", "EPSG:4326"),
        transform=_cube_transform(cube),
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def _apply_nodata(array: np.ndarray, valid_mask: np.ndarray, nodata: float | int, dtype: str) -> np.ndarray:
    result = array.astype(dtype, copy=True)
    result[~valid_mask] = nodata
    return result


def _next_valid_arrays(cube: xr.Dataset, start_index: int) -> dict[str, np.ndarray]:
    shape = (int(cube.sizes["y"]), int(cube.sizes["x"]))
    result = {
        "ndvi": np.full(shape, np.nan, dtype="float32"),
        "ndre": np.full(shape, np.nan, dtype="float32"),
        "gndvi": np.full(shape, np.nan, dtype="float32"),
    }
    filled = np.zeros(shape, dtype=bool)
    quality = cube["quality"].values.astype(bool) if "quality" in cube else np.ones((len(cube.time), *shape), dtype=bool)
    for next_index in range(start_index, len(cube.time)):
        valid = quality[next_index].astype(bool) & np.isfinite(cube["ndvi"].values[next_index]) & ~filled
        if not np.any(valid):
            continue
        for index_name in result:
            values = cube[index_name].values[next_index]
            result[index_name][valid] = values[valid]
        filled |= valid
        if filled.all():
            break
    return result


def _recovery_arrays(
    cube: xr.Dataset,
    event_index: int,
    dates: pd.DatetimeIndex,
    min_days: int,
    max_days: int,
) -> dict[str, np.ndarray]:
    """Return the best later vegetation signal in the expected recovery window."""
    shape = (int(cube.sizes["y"]), int(cube.sizes["x"]))
    result = {
        "ndvi": np.full(shape, np.nan, dtype="float32"),
        "ndre": np.full(shape, np.nan, dtype="float32"),
        "gndvi": np.full(shape, np.nan, dtype="float32"),
    }
    quality = cube["quality"].values.astype(bool) if "quality" in cube else np.ones((len(dates), *shape), dtype=bool)
    event_date = dates[event_index]
    for index in range(event_index + 1, len(dates)):
        day_gap = (dates[index] - event_date).days
        if day_gap > max_days:
            break
        if day_gap < min_days:
            continue
        valid = quality[index] & np.isfinite(cube["ndvi"].values[index])
        for name in result:
            values = cube[name].values[index]
            result[name] = np.where(valid & (np.isnan(result[name]) | (values > result[name])), values, result[name])
    return result


def detect_mowing_rasters_from_cube(
    indices_cube_path: Path,
    output_dir: Path,
    parameters: PlanetTimeSeriesDetectionParameters | None = None,
    grassland_mask_gpkg: Path | None = None,
) -> PlanetRasterDetectionOutputs:
    """Run pixel-wise multi-index mowing detection and save Sentinel-comparable rasters."""
    parameters = parameters or PlanetTimeSeriesDetectionParameters()
    cube = open_dataset(indices_cube_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    ndvi = cube["ndvi"].values
    ndre = cube["ndre"].values
    gndvi = cube["gndvi"].values
    quality = cube["quality"].values.astype(bool) if "quality" in cube else np.isfinite(ndvi)
    dates = pd.to_datetime(cube.time.values).normalize()
    doy = np.array([date.dayofyear for date in dates], dtype="int16")
    height, width = int(cube.sizes["y"]), int(cube.sizes["x"])

    allowed = np.ones((height, width), dtype=bool)
    if grassland_mask_gpkg is not None and grassland_mask_gpkg.exists():
        grasslands = gpd.read_file(grassland_mask_gpkg).to_crs(cube.attrs.get("crs", "EPSG:4326"))
        if grasslands.empty:
            allowed[:] = False
        else:
            allowed = rasterize(
                [(geometry, 1) for geometry in grasslands.geometry if geometry is not None and not geometry.is_empty],
                out_shape=(height, width),
                transform=_cube_transform(cube),
                fill=0,
                dtype="uint8",
            ).astype(bool)

    first_doy = np.zeros((height, width), dtype="int16")
    mowing_doys = np.zeros((6, height, width), dtype="int16")
    last_mowing_doy = np.zeros((height, width), dtype="int16")
    event_count = np.zeros((height, width), dtype="uint8")
    best_confidence = np.zeros((height, width), dtype="float32")
    candidate_events = np.zeros((height, width), dtype="uint8")
    filtered_events = np.zeros((height, width), dtype="uint8")
    overflow_events = np.zeros((height, width), dtype="uint8")
    last_event_doy = np.full((height, width), -9999, dtype="int16")
    event_stack = np.zeros((len(dates), height, width), dtype="uint8")
    valid_observations = (quality & np.isfinite(ndvi)).sum(axis=0).astype("uint16")
    masked_observations = (~quality | ~np.isfinite(ndvi)).sum(axis=0).astype("uint16")
    scl_masked_percent = (masked_observations / len(dates) * 100).astype("float32")

    for time_index in range(1, len(dates)):
        valid_pair = (
            allowed
            & quality[time_index - 1]
            & quality[time_index]
            & np.isfinite(ndvi[time_index - 1])
            & np.isfinite(ndvi[time_index])
            & np.isfinite(ndre[time_index - 1])
            & np.isfinite(ndre[time_index])
            & np.isfinite(gndvi[time_index - 1])
            & np.isfinite(gndvi[time_index])
        )
        if not np.any(valid_pair):
            continue

        ndvi_drop = ndvi[time_index - 1] - ndvi[time_index]
        ndre_drop = ndre[time_index - 1] - ndre[time_index]
        gndvi_drop = gndvi[time_index - 1] - gndvi[time_index]
        support = (
            (ndvi_drop >= parameters.min_ndvi_drop).astype("uint8")
            + (ndre_drop >= parameters.min_ndre_drop).astype("uint8")
            + (gndvi_drop >= parameters.min_gndvi_drop).astype("uint8")
        )
        candidate = (
            valid_pair
            & (support >= parameters.min_indices_with_drop)
            & (ndvi_drop >= parameters.min_ndvi_drop)
            & (ndvi[time_index - 1] >= parameters.min_pre_event_ndvi)
        )
        candidate_events[candidate] = np.minimum(candidate_events[candidate] + 1, np.iinfo("uint8").max)

        next_values = _next_valid_arrays(cube, time_index + 1)
        recovery_ratio = np.divide(
            next_values["ndvi"] - ndvi[time_index],
            ndvi_drop,
            out=np.full((height, width), np.nan, dtype="float32"),
            where=ndvi_drop > 0,
        )
        quick_recovery = (
            (next_values["ndvi"] >= ndvi[time_index - 1] - parameters.max_next_ndvi_tolerance)
            | (recovery_ratio >= parameters.max_quick_recovery_ratio)
        )
        recovery = _recovery_arrays(
            cube,
            time_index,
            dates,
            parameters.min_recovery_days,
            parameters.max_recovery_days,
        )
        recovery_gain = recovery["ndvi"] - ndvi[time_index]
        recovery_ratio = np.divide(
            recovery_gain,
            ndvi_drop,
            out=np.full((height, width), np.nan, dtype="float32"),
            where=ndvi_drop > 0,
        )
        recovered = (
            np.isfinite(recovery["ndvi"])
            & (recovery_gain >= parameters.min_recovery_ndvi_gain)
            & (recovery_ratio >= parameters.min_recovery_ratio)
        )
        sustained = ~quick_recovery & recovered
        filtered = candidate & sustained
        filtered_events[filtered] = np.minimum(filtered_events[filtered] + 1, np.iinfo("uint8").max)
        enough_gap = (last_event_doy < 0) | ((doy[time_index] - last_event_doy) > parameters.deduplicate_days)
        detected = filtered & enough_gap
        if not np.any(detected):
            continue

        confidence = (
            np.clip(ndvi_drop / 0.25, 0, 1) * 0.45
            + np.clip(ndre_drop / 0.12, 0, 1) * 0.25
            + np.clip(gndvi_drop / 0.16, 0, 1) * 0.20
            + (support / 3) * 0.10
            + recovered.astype("float32") * 0.10
        ).astype("float32")
        previous_count = event_count.copy()
        supported_slot = detected & (previous_count < mowing_doys.shape[0])
        overflow = detected & (previous_count >= mowing_doys.shape[0])
        overflow_events[overflow] = 1
        for slot in range(mowing_doys.shape[0]):
            slot_mask = supported_slot & (previous_count == slot)
            mowing_doys[slot, slot_mask] = doy[time_index]
        first_doy[(first_doy == 0) & detected] = doy[time_index]
        last_mowing_doy[detected] = doy[time_index]
        event_count[detected] = np.minimum(event_count[detected] + 1, np.iinfo("uint8").max)
        best_confidence[detected] = np.maximum(best_confidence[detected], confidence[detected])
        last_event_doy[detected] = doy[time_index]
        event_stack[time_index, detected] = 1

    intervals = {}
    for slot in range(5):
        interval = np.zeros((height, width), dtype="int16")
        valid_interval = (mowing_doys[slot] > 0) & (mowing_doys[slot + 1] > 0)
        interval[valid_interval] = mowing_doys[slot + 1, valid_interval] - mowing_doys[slot, valid_interval]
        intervals[f"INTERVAL_{slot + 1}_{slot + 2}.tif"] = interval

    grassland_valid = allowed
    quality_flag = np.zeros((height, width), dtype="uint8")
    quality_flag[grassland_valid & (valid_observations > 0)] = 1
    quality_flag[grassland_valid & (valid_observations >= 20) & (scl_masked_percent <= 70)] = 2
    quality_flag[grassland_valid & (valid_observations >= 40) & (scl_masked_percent <= 50)] = 3

    rasters: dict[str, Path] = {}

    def save(name: str, array: np.ndarray, dtype: str, nodata: float | int, valid_mask: np.ndarray | None = None) -> None:
        path = output_dir / name
        values = _apply_nodata(array, valid_mask, nodata, dtype) if valid_mask is not None else array
        _write_raster(path, values, cube, dtype, nodata)
        rasters[name] = path

    save("GRASSLAND_MASK.tif", allowed.astype("uint8"), "uint8", 0)
    save("MOWING_COUNT.tif", event_count, "uint8", 255, grassland_valid)
    for slot in range(6):
        save(f"MOWING_DOY_{slot + 1}.tif", mowing_doys[slot], "int16", -9999, grassland_valid)
    save("LAST_MOWING_DOY.tif", last_mowing_doy, "int16", -9999, grassland_valid)
    save("MOWING_CONFIDENCE.tif", best_confidence, "float32", -9999, grassland_valid)
    save("VALID_OBSERVATIONS.tif", valid_observations, "uint16", 65535, grassland_valid)
    save("SCL_MASKED_PERCENT.tif", scl_masked_percent, "float32", -9999, grassland_valid)
    save("QUALITY_FLAG.tif", quality_flag, "uint8", 0)
    save("CANDIDATE_EVENTS.tif", candidate_events, "uint8", 255, grassland_valid)
    save("FILTERED_EVENTS.tif", filtered_events, "uint8", 255, grassland_valid)
    save("OVERFLOW_EVENTS.tif", overflow_events, "uint8", 255, grassland_valid)
    for name, array in intervals.items():
        save(name, array, "int16", -9999, grassland_valid)

    events_nc_path = output_dir / "MOWING_EVENT_STACK.nc"

    event_cube = xr.Dataset(
        {"mowing_event": (("time", "y", "x"), event_stack)},
        coords={"time": cube.time.values, "y": cube.y.values, "x": cube.x.values},
        attrs={**cube.attrs, "description": "Pixel-wise PlanetScope mowing detections. 1 = mowing event."},
    )
    event_cube.to_netcdf(events_nc_path)

    return PlanetRasterDetectionOutputs(
        output_dir=output_dir,
        rasters=rasters,
        events_nc=events_nc_path,
    )


def detect_mowing_rasters_sgs_from_cube(
    indices_cube_path: Path,
    output_dir: Path,
    parameters: PlanetTimeSeriesDetectionParameters | None = None,
    grassland_mask_gpkg: Path | None = None,
) -> PlanetRasterDetectionOutputs:
    """Run a separate Sentinel-compatible SGS detector on every valid pixel."""
    parameters = parameters or PlanetTimeSeriesDetectionParameters()
    cube = open_dataset(indices_cube_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ndvi = cube["ndvi"].values.astype("float32")
    quality = cube["quality"].values.astype(bool) if "quality" in cube else np.isfinite(ndvi)
    dates = pd.DatetimeIndex(pd.to_datetime(cube.time.values).normalize())
    height, width = int(cube.sizes["y"]), int(cube.sizes["x"])
    allowed = np.ones((height, width), dtype=bool)
    if grassland_mask_gpkg is not None and grassland_mask_gpkg.exists():
        grasslands = gpd.read_file(grassland_mask_gpkg).to_crs(cube.attrs.get("crs", "EPSG:4326"))
        allowed = rasterize(
            [(geometry, 1) for geometry in grasslands.geometry if geometry is not None and not geometry.is_empty],
            out_shape=(height, width), transform=_cube_transform(cube), fill=0, dtype="uint8",
        ).astype(bool) if not grasslands.empty else np.zeros((height, width), dtype=bool)
    first_doy = np.zeros((height, width), dtype="int16")
    mowing_doys = np.zeros((6, height, width), dtype="int16")
    event_count = np.zeros((height, width), dtype="uint8")
    confidence = np.zeros((height, width), dtype="float32")
    valid_observations = (quality & np.isfinite(ndvi)).sum(axis=0).astype("uint16")
    events_stack = np.zeros((len(dates), height, width), dtype="uint8")
    for row, col in zip(*np.where(allowed)):
        values = ndvi[:, row, col].copy()
        values[~quality[:, row, col]] = np.nan
        events = _detect_sgs_events(dates, values, parameters)
        for slot, (date_index, drop) in enumerate(events[:6]):
            mowing_doys[slot, row, col] = dates[date_index].dayofyear
            events_stack[date_index, row, col] = 1
            confidence[row, col] = max(confidence[row, col], min(1.0, drop / 0.25))
        if events:
            event_count[row, col] = min(len(events), mowing_doys.shape[0])
            first_doy[row, col] = mowing_doys[0, row, col]
    grassland_valid = allowed
    rasters: dict[str, Path] = {}
    def save(name: str, array: np.ndarray, dtype: str, nodata: float | int):
        path = output_dir / name
        _write_raster(path, _apply_nodata(array, grassland_valid, nodata, dtype), cube, dtype, nodata)
        rasters[name] = path
    save("GRASSLAND_MASK.tif", allowed.astype("uint8"), "uint8", 0)
    save("MOWING_COUNT.tif", event_count, "uint8", 255)
    for slot in range(6):
        save(f"MOWING_DOY_{slot + 1}.tif", mowing_doys[slot], "int16", -9999)
    save("MOWING_CONFIDENCE.tif", confidence, "float32", -9999)
    save("VALID_OBSERVATIONS.tif", valid_observations, "uint16", 65535)
    events_nc = output_dir / "MOWING_EVENT_STACK.nc"
    xr.Dataset(
        {"mowing_event": (("time", "y", "x"), events_stack)},
        coords={"time": cube.time.values, "y": cube.y.values, "x": cube.x.values},
        attrs={**cube.attrs, "description": "Sentinel-compatible SGS PlanetScope mowing detections."},
    ).to_netcdf(events_nc)
    return PlanetRasterDetectionOutputs(output_dir, rasters, events_nc)


def detect_mowing_rasters_final_from_cube(
    indices_cube_path: Path,
    output_dir: Path,
    parameters: PlanetTimeSeriesDetectionParameters | None = None,
    grassland_mask_gpkg: Path | None = None,
    min_spatial_support: int | None = None,
    neighbour_date_tolerance: int | None = None,
) -> PlanetRasterDetectionOutputs:
    """SGS detector with multi-index and spatial-support filtering."""
    parameters = parameters or PlanetTimeSeriesDetectionParameters()
    min_spatial_support = min_spatial_support or parameters.final_min_component_pixels
    neighbour_date_tolerance = neighbour_date_tolerance or parameters.final_neighbour_date_tolerance_days
    cube = open_dataset(indices_cube_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.DatetimeIndex(pd.to_datetime(cube.time.values).normalize())
    detection_start = pd.Timestamp(
        year=int(dates[0].year),
        month=parameters.detection_start_month,
        day=parameters.detection_start_day,
    )
    detection_end = pd.Timestamp(
        year=int(dates[0].year),
        month=parameters.detection_end_month,
        day=parameters.detection_end_day,
    )
    detection_window = (dates >= detection_start) & (dates <= detection_end)
    ndvi, ndre, gndvi = (cube[name].values.astype("float32") for name in ("ndvi", "ndre", "gndvi"))
    quality = cube["quality"].values.astype(bool) if "quality" in cube else np.isfinite(ndvi)
    height, width = int(cube.sizes["y"]), int(cube.sizes["x"])
    allowed = np.ones((height, width), dtype=bool)
    if grassland_mask_gpkg is not None and grassland_mask_gpkg.exists():
        grasslands = gpd.read_file(grassland_mask_gpkg).to_crs(cube.attrs.get("crs", "EPSG:4326"))
        allowed = rasterize([(g, 1) for g in grasslands.geometry if g is not None and not g.is_empty], out_shape=(height, width), transform=_cube_transform(cube), fill=0, dtype="uint8").astype(bool)

    pixel_events: list[list[tuple[int, int, int, float]]] = [[] for _ in range(height * width)]
    for row, col in zip(*np.where(allowed)):
        values = ndvi[:, row, col].copy()
        values[~quality[:, row, col]] = np.nan
        pixel_events[row * width + col] = _detect_sgs_event_details(dates, values, parameters)

    accepted = np.zeros((len(dates), height, width), dtype=bool)
    for date_index in range(len(dates)):
        if not detection_window[date_index]:
            continue
        candidate = np.zeros((height, width), dtype=bool)
        for row, col in zip(*np.where(allowed)):
            if any(event_index == date_index for event_index, _, _, _ in pixel_events[row * width + col]):
                candidate[row, col] = True
        if not candidate.any():
            continue
        supported = np.zeros_like(candidate)
        for row, col in zip(*np.where(candidate)):
            r0, r1 = max(0, row - 1), min(height, row + 2)
            c0, c1 = max(0, col - 1), min(width, col + 2)
            neighbours = 0
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if allowed[rr, cc] and any(abs(dates[event_index] - dates[date_index]).days <= neighbour_date_tolerance for event_index, _, _, _ in pixel_events[rr * width + cc]):
                        neighbours += 1
            supported[row, col] = neighbours >= parameters.final_min_neighbour_events
        components, count = connected_components(supported, structure=np.ones((3, 3), dtype=np.uint8))
        sizes = np.bincount(components.ravel())
        spatial = supported & (sizes[components] >= min_spatial_support)
        for row, col in zip(*np.where(spatial)):
            matching = [
                (peak_index, recovery_index, drop)
                for event_index, peak_index, recovery_index, drop in pixel_events[row * width + col]
                if event_index == date_index
            ]
            if not matching:
                continue
            event_index = date_index
            peak_index, recovery_index, ndvi_drop = max(matching, key=lambda item: item[2])
            valid = quality[peak_index, row, col] & quality[event_index, row, col]
            if not valid:
                continue
            drops = [
                ndvi[peak_index, row, col] - ndvi[event_index, row, col],
                ndre[peak_index, row, col] - ndre[event_index, row, col],
                gndvi[peak_index, row, col] - gndvi[event_index, row, col],
            ]
            support = sum(drop >= threshold for drop, threshold in zip(drops, (parameters.min_ndvi_drop, parameters.min_ndre_drop, parameters.min_gndvi_drop)))
            recovered = (
                quality[recovery_index, row, col]
                and np.isfinite(ndvi[recovery_index, row, col])
                and ndvi[recovery_index, row, col] - ndvi[event_index, row, col] >= parameters.min_recovery_ndvi_gain
            )
            if support >= parameters.min_indices_with_drop and recovered and ndvi_drop >= parameters.sgs_drop_threshold:
                accepted[date_index, row, col] = True

    count = accepted.sum(axis=0).astype("uint8")
    candidate_count = np.zeros((height, width), dtype="uint8")
    doys = np.zeros((6, height, width), dtype="int16")
    last_doy = np.zeros((height, width), dtype="int16")
    confidence = np.zeros((height, width), dtype="float32")
    for row, col in zip(*np.where(allowed)):
        candidate_count[row, col] = min(len(pixel_events[row * width + col]), 255)
        event_indices = np.where(accepted[:, row, col])[0]
        for slot, event_index in enumerate(event_indices[:6]):
            doys[slot, row, col] = dates[event_index].dayofyear
        if len(event_indices):
            last_doy[row, col] = dates[event_indices[-1]].dayofyear
            confidence[row, col] = min(1.0, float(count[row, col]) / 3.0)
    intervals = {}
    for slot in range(5):
        interval = np.zeros((height, width), dtype="int16")
        valid_interval = (doys[slot] > 0) & (doys[slot + 1] > 0)
        interval[valid_interval] = doys[slot + 1, valid_interval] - doys[slot, valid_interval]
        intervals[f"INTERVAL_{slot + 1}_{slot + 2}.tif"] = interval
    valid_observations = (quality & np.isfinite(ndvi)).sum(axis=0).astype("uint16")
    masked_observations = (~quality | ~np.isfinite(ndvi)).sum(axis=0).astype("uint16")
    scl_masked_percent = (masked_observations / len(dates) * 100).astype("float32")
    quality_flag = np.zeros((height, width), dtype="uint8")
    quality_flag[allowed & (valid_observations > 0)] = 1
    quality_flag[allowed & (valid_observations >= 20) & (scl_masked_percent <= 70)] = 2
    quality_flag[allowed & (valid_observations >= 40) & (scl_masked_percent <= 50)] = 3
    overflow = (accepted.sum(axis=0) > 6).astype("uint8")
    rasters: dict[str, Path] = {}
    def save(name: str, array: np.ndarray, dtype: str, nodata: float | int):
        path = output_dir / name
        _write_raster(path, _apply_nodata(array, allowed, nodata, dtype), cube, dtype, nodata)
        rasters[name] = path
    save("GRASSLAND_MASK.tif", allowed.astype("uint8"), "uint8", 0)
    save("MOWING_COUNT.tif", count, "uint8", 255)
    for slot in range(6):
        save(f"MOWING_DOY_{slot + 1}.tif", doys[slot], "int16", -9999)
    save("LAST_MOWING_DOY.tif", last_doy, "int16", -9999)
    save("MOWING_CONFIDENCE.tif", confidence, "float32", -9999)
    save("VALID_OBSERVATIONS.tif", valid_observations, "uint16", 65535)
    save("SCL_MASKED_PERCENT.tif", scl_masked_percent, "float32", -9999)
    save("QUALITY_FLAG.tif", quality_flag, "uint8", 0)
    save("CANDIDATE_EVENTS.tif", candidate_count, "uint8", 255)
    save("FILTERED_EVENTS.tif", count, "uint8", 255)
    save("OVERFLOW_EVENTS.tif", overflow, "uint8", 255)
    for name, array in intervals.items():
        save(name, array, "int16", -9999)
    events_nc = output_dir / "MOWING_EVENT_STACK.nc"
    xr.Dataset({"mowing_event": (("time", "y", "x"), accepted.astype("uint8"))}, coords={"time": cube.time.values, "y": cube.y.values, "x": cube.x.values}, attrs={**cube.attrs, "description": "Filtered multi-index and spatial-support PlanetScope mowing detections."}).to_netcdf(events_nc)
    return PlanetRasterDetectionOutputs(output_dir, rasters, events_nc)


def detect_mowing_from_candidate_timeseries(
    indices_cube_path: Path,
    candidates_csv: Path,
    output_csv: Path,
    output_gpkg: Path | None = None,
    parameters: PlanetTimeSeriesDetectionParameters | None = None,
) -> pd.DataFrame:
    """Detect mowing events from candidate polygons and multi-index time-series shape."""
    parameters = parameters or PlanetTimeSeriesDetectionParameters()
    cube = open_dataset(indices_cube_path)
    candidates = pd.read_csv(candidates_csv)
    if candidates.empty:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(output_csv, index=False)
        return candidates

    dates = pd.to_datetime(cube.time.values).normalize()
    date_to_index = {date: idx for idx, date in enumerate(dates)}
    rows: list[dict[str, object]] = []

    for candidate in candidates.itertuples(index=False):
        row = pd.Series(candidate._asdict())
        if float(row.get("area_pixels", 0)) < parameters.min_area_pixels:
            continue

        event_date = pd.Timestamp(row["event_date"]).normalize()
        previous_date = pd.Timestamp(row["previous_date"]).normalize()
        event_idx = date_to_index.get(event_date)
        previous_idx = date_to_index.get(previous_date)
        if event_idx is None or previous_idx is None:
            continue

        index_features: dict[str, float | str | bool] = {}
        drops = []
        valid_fractions = []
        quick_recovery = False
        sustained = True
        recovered = False

        for index_name in ("ndvi", "ndre", "gndvi"):
            values, valid = _series_for_polygon(cube, row, index_name)
            previous_value = values[previous_idx]
            event_value = values[event_idx]
            next_idx = _next_valid_index(values, valid, event_idx + 1, parameters.minimum_valid_fraction)
            next_value = values[next_idx] if next_idx is not None else np.nan
            drop = previous_value - event_value
            recovery_ratio = ((next_value - event_value) / drop) if np.isfinite(next_value) and drop > 0 else np.nan

            index_features[f"{index_name}_previous"] = previous_value
            index_features[f"{index_name}_event"] = event_value
            index_features[f"{index_name}_next"] = next_value
            index_features[f"{index_name}_drop"] = drop
            index_features[f"{index_name}_recovery_ratio"] = recovery_ratio
            index_features[f"{index_name}_valid_previous"] = valid[previous_idx]
            index_features[f"{index_name}_valid_event"] = valid[event_idx]
            index_features[f"{index_name}_valid_next"] = valid[next_idx] if next_idx is not None else np.nan

            drops.append(drop)
            valid_fractions.extend([valid[previous_idx], valid[event_idx]])
            if index_name == "ndvi":
                index_features["next_valid_date"] = dates[next_idx].date().isoformat() if next_idx is not None else ""
                quick_recovery = bool(
                    next_idx is not None
                    and (
                        next_value >= previous_value - parameters.max_next_ndvi_tolerance
                        or recovery_ratio >= parameters.max_quick_recovery_ratio
                    )
                )
                sustained = bool(next_idx is None or next_value < previous_value - parameters.max_next_ndvi_tolerance)

                recovery_values = []
                for future_idx in range(event_idx + 1, len(dates)):
                    day_gap = (dates[future_idx] - event_date).days
                    if day_gap > parameters.max_recovery_days:
                        break
                    if day_gap >= parameters.min_recovery_days and valid[future_idx] and np.isfinite(values[future_idx]):
                        recovery_values.append(values[future_idx])
                if recovery_values:
                    recovery_value = max(recovery_values)
                    recovery_gain = recovery_value - event_value
                    recovery_ratio = recovery_gain / drop if drop > 0 else np.nan
                    recovered = bool(
                        recovery_gain >= parameters.min_recovery_ndvi_gain
                        and recovery_ratio >= parameters.min_recovery_ratio
                    )
                index_features["recovery_ndvi"] = recovery_values[-1] if recovery_values else np.nan
                index_features["recovered_in_window"] = recovered

        ndvi_drop = float(index_features["ndvi_drop"])
        ndvi_previous = float(index_features["ndvi_previous"])
        ndre_drop = float(index_features["ndre_drop"])
        gndvi_drop = float(index_features["gndvi_drop"])
        index_support = int(
            (ndvi_drop >= parameters.min_ndvi_drop)
            + (ndre_drop >= parameters.min_ndre_drop)
            + (gndvi_drop >= parameters.min_gndvi_drop)
        )
        valid_enough = min(valid_fractions) >= parameters.minimum_valid_fraction
        detected = bool(
            valid_enough
            and index_support >= parameters.min_indices_with_drop
            and ndvi_drop >= parameters.min_ndvi_drop
            and sustained
            and not quick_recovery
            and np.isfinite(ndvi_previous)
            and ndvi_previous >= parameters.min_pre_event_ndvi
            and recovered
        )
        if not detected:
            continue

        score = float(
            np.clip(ndvi_drop / 0.25, 0, 1) * 0.45
            + np.clip(ndre_drop / 0.12, 0, 1) * 0.25
            + np.clip(gndvi_drop / 0.16, 0, 1) * 0.20
            + (index_support / 3) * 0.10
        )
        polygon = _pixel_polygon(row)
        centroid = polygon.centroid
        segment_key = f"{round(centroid.x / 6):04.0f}_{round(centroid.y / 6):04.0f}"

        rows.append(
            {
                **row.to_dict(),
                **index_features,
                "index_support": index_support,
                "quick_recovery": quick_recovery,
                "sustained_drop": sustained,
                "score": score,
                "segment_key": segment_key,
            }
        )

    detected_events = _deduplicate_events(pd.DataFrame(rows), parameters.deduplicate_days)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    detected_events.to_csv(output_csv, index=False)

    if output_gpkg is not None and not detected_events.empty:
        geo_rows = detected_events.copy()
        geo_rows["geometry"] = [
            _pixel_to_geo_polygon(cube, _pixel_polygon(pd.Series(row)))
            for row in geo_rows.to_dict("records")
        ]
        gdf = gpd.GeoDataFrame(geo_rows, geometry="geometry", crs="EPSG:4326")
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)
        if output_gpkg.exists():
            output_gpkg.unlink()
        gdf.to_file(output_gpkg, driver="GPKG")

    return detected_events
