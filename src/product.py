"""Small end-user workflow for PlanetScope mowing products.

The module deliberately composes the existing cube, index and detector code.
It does not contain a second mowing algorithm.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from IPython.display import display
from ipywidgets import Button, Dropdown, HBox, HTML, Textarea, VBox
from matplotlib.patches import Rectangle
from scipy.ndimage import label

from .collection import build_collection_cube
from .config import WorkflowConfig
from .download import download_planetscope
from .indices import build_indices
from .io import open_dataset
from .timeseries_detection import (
    PlanetTimeSeriesDetectionParameters,
    detect_mowing_rasters_final_from_cube,
)


def _default_product_detection_parameters() -> PlanetTimeSeriesDetectionParameters:
    """Use the same tuned detector settings as notebook 05c."""
    return PlanetTimeSeriesDetectionParameters(
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
        final_min_component_pixels=5,
        final_neighbour_date_tolerance_days=7,
        detection_start_month=4,
        detection_start_day=1,
        detection_end_month=10,
        detection_end_day=15,
    )


def run_product(
    workflow: WorkflowConfig,
    *,
    planet_api_key: str | None = None,
    collection_id: str | None = None,
    sentinelhub_client_id: str | None = None,
    sentinelhub_client_secret: str | None = None,
    resolution: float = 3.0,
    grassland_mask: Path | None = None,
    parameters: PlanetTimeSeriesDetectionParameters | None = None,
    force_rebuild: bool = False,
    rebuild_planet_cube: bool = False,
    rebuild_indices_cube: bool = False,
    rebuild_detection: bool = False,
) -> dict[str, Path]:
    """Prepare a complete area product and return its main output paths.

    Existing cubes, index cubes and detections are reused unless their rebuild
    flag is true. ``force_rebuild`` remains as a backwards-compatible alias for
    rebuilding everything.
    Set ``collection_id`` to use an existing Planet/Sentinel Hub BYOC
    collection; otherwise ``planet_api_key`` is used through the existing
    Planet Orders workflow.
    """
    workflow.create_directories()
    rebuild_planet_cube = force_rebuild or rebuild_planet_cube
    rebuild_indices_cube = force_rebuild or rebuild_indices_cube
    rebuild_detection = force_rebuild or rebuild_detection

    if rebuild_planet_cube or not workflow.planet_cube_path.exists():
        if collection_id:
            build_collection_cube(
                workflow,
                collection_id=collection_id,
                client_id=sentinelhub_client_id,
                client_secret=sentinelhub_client_secret,
                resolution=resolution,
            )
        else:
            if not planet_api_key:
                raise ValueError("Provide planet_api_key or collection_id.")
            download_planetscope(workflow, planet_api_key)

    if rebuild_indices_cube or not workflow.indices_cube_path.exists():
        build_indices(workflow.planet_cube_path, workflow.indices_cube_path)

    detection_dir = workflow.outputs_dir / "product_detection"
    count_path = detection_dir / "MOWING_COUNT.tif"
    if rebuild_detection or not count_path.exists():
        detect_mowing_rasters_final_from_cube(
            workflow.indices_cube_path,
            detection_dir,
            parameters=parameters or _default_product_detection_parameters(),
            grassland_mask_gpkg=grassland_mask,
        )

    catalog_path = build_event_catalog(
        workflow.indices_cube_path,
        detection_dir,
        workflow.outputs_dir / "events" / "event_catalog.csv",
    )
    return {
        "planet_cube": workflow.planet_cube_path,
        "indices_cube": workflow.indices_cube_path,
        "detection_dir": detection_dir,
        "event_catalog": catalog_path,
    }


def _event_stack_path(detection_dir: Path) -> Path:
    path = detection_dir / "MOWING_EVENT_STACK.nc"
    if not path.exists():
        raise FileNotFoundError(f"Detection event stack not found: {path}")
    return path


def build_event_catalog(
    indices_cube_path: Path,
    detection_dir: Path,
    output_csv: Path | None = None,
    min_component_pixels: int = 10,
) -> Path:
    """Create one row per date/component and per-date event rasters.

    The event raster is binary: 1 means that the detector accepted a mowing
    event for that pixel on that date. Components make the review unit easy to
    select while preserving the underlying pixel-level detections.
    """
    cube = open_dataset(indices_cube_path)
    stack = xr.open_dataset(_event_stack_path(detection_dir))["mowing_event"].values.astype(bool)
    dates = pd.DatetimeIndex(pd.to_datetime(cube.time.values).normalize())
    event_dir = (output_csv.parent if output_csv else detection_dir.parent / "events") / "by_date"
    event_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for time_index, date in enumerate(dates):
        event_mask = stack[time_index]
        date_path = event_dir / f"MOWING_{date.date().isoformat()}.tif"
        with rasterio.open(detection_dir / "MOWING_COUNT.tif") as src:
            profile = src.profile.copy()
            profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
            with rasterio.open(date_path, "w", **profile) as dst:
                dst.write(event_mask.astype("uint8"), 1)
        components, component_count = label(event_mask, structure=np.ones((3, 3), dtype=np.uint8))
        for component_id in range(1, component_count + 1):
            yy, xx = np.where(components == component_id)
            if len(xx) < min_component_pixels:
                continue
            rows.append(
                {
                    "event_id": f"{date.date().isoformat()}_{component_id:04d}",
                    "event_date": date.date().isoformat(),
                    "event_index": time_index,
                    "component_id": component_id,
                    "pixel_count": len(xx),
                    "row_min": int(yy.min()),
                    "row_max": int(yy.max()),
                    "col_min": int(xx.min()),
                    "col_max": int(xx.max()),
                    "center_row": float(yy.mean()),
                    "center_col": float(xx.mean()),
                    "event_raster": str(date_path),
                }
            )
    catalog = pd.DataFrame(rows)
    target = output_csv or detection_dir.parent / "events" / "event_catalog.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(target, index=False)
    return target


def review_events(
    indices_cube_path: Path,
    planet_cube_path: Path,
    detection_dir: Path,
    catalog_csv: Path,
    output_csv: Path,
) -> None:
    """Interactive date/event reviewer for the product output."""
    indices = open_dataset(indices_cube_path)
    planet = open_dataset(planet_cube_path)
    catalog = pd.read_csv(catalog_csv)
    if catalog.empty:
        display(HTML("No detected mowing events were found."))
        return
    catalog["event_date"] = pd.to_datetime(catalog["event_date"])
    catalog = catalog.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    dates = sorted(catalog["event_date"].dt.date.astype(str).unique())
    date_selector = Dropdown(options=dates, description="Date")
    event_selector = Dropdown(description="Event")
    previous_event = Button(description="Previous event")
    next_event = Button(description="Next event")
    accept = Button(description="Mowing", button_style="success")
    reject = Button(description="Not mowing", button_style="danger")
    uncertain = Button(description="Uncertain")
    note = Textarea(placeholder="Optional note", description="Note")
    status = HTML("Select a date and event.")
    state: dict[str, object] = {"row": None, "syncing": False}

    dates_index = pd.DatetimeIndex(pd.to_datetime(indices.time.values).normalize())
    date_to_index = {date.date().isoformat(): index for index, date in enumerate(dates_index)}
    max_bad_pixel_fraction = 0.30
    bad_pixel_fraction = np.zeros(len(dates_index), dtype="float32")
    if "quality" in planet:
        quality = planet["quality"].values
        bad_pixel_fraction = np.mean(~np.isfinite(quality) | (quality == 0), axis=(1, 2)).astype("float32")
    else:
        ndvi_values = indices["ndvi"].values
        bad_pixel_fraction = np.mean(~np.isfinite(ndvi_values), axis=(1, 2)).astype("float32")
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1, 1.15))
    image_axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    ts_ax = fig.add_subplot(grid[1, :])

    def event_options(_: object | None = None) -> None:
        if state.get("syncing"):
            return
        selected = catalog[catalog["event_date"].dt.date.astype(str) == date_selector.value]
        event_selector.options = [
            (f"{row.event_id} | {int(row.pixel_count)} pixels", int(index))
            for index, row in selected.iterrows()
        ]
        if event_selector.options:
            event_selector.value = event_selector.options[0][1]
            refresh()

    def select_event_by_catalog_index(index: int) -> None:
        index = max(0, min(len(catalog) - 1, index))
        event_date = catalog.loc[index, "event_date"].date().isoformat()
        state["syncing"] = True
        try:
            if date_selector.value != event_date:
                date_selector.value = event_date
            selected = catalog[catalog["event_date"].dt.date.astype(str) == event_date]
            event_selector.options = [
                (f"{row.event_id} | {int(row.pixel_count)} pixels", int(row_index))
                for row_index, row in selected.iterrows()
            ]
            event_selector.value = index
        finally:
            state["syncing"] = False
        refresh()

    def move_event(step: int) -> None:
        if event_selector.value is None:
            select_event_by_catalog_index(0)
            return
        current = int(event_selector.value)
        select_event_by_catalog_index(current + step)

    def image(cube: xr.Dataset, name: str, time_index: int, row: pd.Series | None) -> np.ndarray:
        if name in cube:
            values = cube[name].isel(time=time_index).values.astype("float32")
        else:
            values = np.zeros((cube.sizes["y"], cube.sizes["x"]), dtype="float32")
        finite = np.isfinite(values)
        if finite.any():
            low, high = np.nanpercentile(values[finite], [2, 98])
            values = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
        if row is not None:
            mask = np.zeros_like(values, dtype=bool)
            mask[int(row.row_min): int(row.row_max) + 1, int(row.col_min): int(row.col_max) + 1] = True
            values = np.ma.masked_where(~mask, values)
        return values

    def display_rgb(cube: xr.Dataset, time_index: int) -> np.ndarray:
        """Create an RGB display image; fall back gracefully if blue is absent."""
        if {"red", "green", "blue"}.issubset(cube.data_vars):
            channel_names = ("red", "green", "blue")
        else:
            channel_names = ("red", "green", "nir")
        channels = []
        for name in channel_names:
            values = cube[name].isel(time=time_index).values.astype("float32")
            finite = np.isfinite(values)
            if finite.any():
                low, high = np.nanpercentile(values[finite], [2, 98])
                values = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
            else:
                values = np.zeros_like(values)
            channels.append(values)
        return np.dstack(channels)

    def nearest_clear_index(start_index: int, direction: int) -> int:
        index = start_index + direction
        while 0 <= index < len(dates_index):
            if bad_pixel_fraction[index] <= max_bad_pixel_fraction:
                return index
            index += direction
        return max(0, min(len(dates_index) - 1, start_index + direction))

    def refresh(_: object | None = None) -> None:
        if event_selector.value is None:
            return
        row = catalog.loc[int(event_selector.value)]
        event_date = row.event_date.date().isoformat()
        event_index = date_to_index.get(event_date, int(row.event_index))
        previous_index = nearest_clear_index(event_index, -1)
        state["row"] = row
        for ax in image_axes:
            ax.clear()
        next_index = nearest_clear_index(event_index, 1)
        previous_rgb = display_rgb(planet, previous_index)
        current_rgb = display_rgb(planet, event_index)
        following_rgb = display_rgb(planet, next_index)
        delta = indices["ndvi"].isel(time=event_index).values - indices["ndvi"].isel(time=previous_index).values
        delta = np.ma.masked_invalid(delta)
        daily_event_mask = np.zeros((indices.sizes["y"], indices.sizes["x"]), dtype=bool)
        selected_event_mask = np.zeros((indices.sizes["y"], indices.sizes["x"]), dtype=bool)
        event_raster = Path(str(row.event_raster))
        if not event_raster.exists():
            # Catalogs may contain paths from the machine that created them.
            # Try both layouts used by the product and the simple reviewer.
            candidates = [
                detection_dir / "events" / "by_date" / event_raster.name,
                detection_dir.parent / "events" / "by_date" / event_raster.name,
            ]
            event_raster = next(
                (candidate for candidate in candidates if candidate.exists()),
                candidates[0],
            )
        if event_raster.exists():
            with rasterio.open(event_raster) as src:
                raw_event_mask = src.read(1).astype(bool)
            components, _ = label(raw_event_mask, structure=np.ones((3, 3), dtype=np.uint8))
            daily_rows = catalog[catalog["event_date"].dt.date.astype(str) == event_date]
            daily_component_ids = {
                int(component_id)
                for component_id in daily_rows["component_id"].dropna().values
            }
            for component_id in daily_component_ids:
                daily_event_mask |= components == component_id
            selected_event_mask = components == int(row.component_id)
        resolution = float(planet.attrs.get("resolution", 3.0))
        zoom_margin_pixels = max(1, int(round(100 / resolution)))
        zoom_col_min = max(0, int(row.col_min) - zoom_margin_pixels)
        zoom_col_max = min(indices.sizes["x"] - 1, int(row.col_max) + zoom_margin_pixels)
        zoom_row_min = max(0, int(row.row_min) - zoom_margin_pixels)
        zoom_row_max = min(indices.sizes["y"] - 1, int(row.row_max) + zoom_margin_pixels)
        image_axes[0].imshow(previous_rgb, origin="upper")
        image_axes[0].set_title(f"{dates_index[previous_index].date()}")
        image_axes[1].imshow(current_rgb, origin="upper")
        image_axes[1].imshow(
            np.ma.masked_where(~daily_event_mask, daily_event_mask),
            cmap="autumn",
            alpha=0.35,
            origin="upper",
        )
        image_axes[1].contour(daily_event_mask.astype("uint8"), levels=[0.5], colors="orange", linewidths=0.8)
        image_axes[1].contour(selected_event_mask.astype("uint8"), levels=[0.5], colors="yellow", linewidths=1.8)
        image_axes[1].set_title(f"{dates_index[event_index].date()}")
        image_axes[2].imshow(following_rgb, origin="upper")
        image_axes[2].set_title(f"{dates_index[next_index].date()}")
        image_axes[3].imshow(delta, cmap="RdBu_r", origin="upper", vmin=-0.35, vmax=0.35)
        image_axes[3].set_title("Delta NDVI: dogodek - prej")
        for ax in image_axes[:3]:
            ax.add_patch(
                Rectangle(
                    (row.col_min, row.row_min),
                    row.col_max - row.col_min + 1,
                    row.row_max - row.row_min + 1,
                    fill=False,
                    edgecolor="yellow",
                    linewidth=1.5,
                )
            )
        for ax in image_axes:
            ax.set_xlim(zoom_col_min, zoom_col_max)
            ax.set_ylim(zoom_row_max, zoom_row_min)
            ax.set_axis_off()
        ts_ax.clear()
        center_row = int(round(row.center_row))
        center_col = int(round(row.center_col))
        for name, color in (("ndvi", "tab:blue"), ("ndre", "tab:orange"), ("gndvi", "tab:green")):
            values = indices[name].isel(y=center_row, x=center_col).values
            finite = np.isfinite(values)
            ts_ax.plot(dates_index[finite], values[finite], marker="o", ms=3, color=color, label=name.upper())
        ts_ax.axvline(dates_index[event_index], color="red", linestyle="--", label="detected")
        ts_ax.set_title(f"Pixel time series | x={center_col}, y={center_row}")
        ts_ax.set_ylabel("Index")
        ts_ax.grid(alpha=0.3)
        ts_ax.legend()
        fig.autofmt_xdate()
        position = int(event_selector.value) + 1
        date_events = int((catalog["event_date"].dt.date.astype(str) == event_date).sum())
        status.value = (
            f"Dogodek {position}/{len(catalog)} | datum {event_date} | "
            f"{date_events} dogodkov na ta datum | "
            "oranžno = zaznana košnja. Izberi Mowing, Not mowing ali Uncertain."
        )
        # draw() is more reliable than draw_idle() in the VS Code notebook backend.
        fig.canvas.draw()

    def save_review(label_name: str) -> None:
        row = state.get("row")
        if row is None:
            status.value = "Najprej izberi dogodek."
            return
        record = pd.DataFrame([{
            "review_timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
            "event_id": row.event_id,
            "event_date": row.event_date.date().isoformat(),
            "pixel_count": row.pixel_count,
            "center_row": row.center_row,
            "center_col": row.center_col,
            "review_label": label_name,
            "note": note.value,
        }])
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        record.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)
        note.value = ""
        status.value = f"Shranjeno: {row.event_id} = {label_name}"
        move_event(1)

    date_selector.observe(event_options, names="value")
    event_selector.observe(refresh, names="value")
    previous_event.on_click(lambda _: move_event(-1))
    next_event.on_click(lambda _: move_event(1))
    accept.on_click(lambda _: save_review("mowing"))
    reject.on_click(lambda _: save_review("not_mowing"))
    uncertain.on_click(lambda _: save_review("uncertain"))
    event_options()
    display(VBox([
        HBox([date_selector, event_selector, previous_event, next_event]),
        HBox([accept, reject, uncertain]),
        note,
        status,
    ]))
    refresh()


def evaluate_reviews(review_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Summarise reviewed detections; this measures precision, not recall."""
    reviews = pd.read_csv(review_csv)
    if reviews.empty:
        summary = pd.DataFrame([{"reviewed_events": 0, "confirmed_mowing": 0, "confirmed_not_mowing": 0, "uncertain": 0, "precision_among_decided": np.nan}])
    else:
        counts = reviews["review_label"].value_counts()
        mowing = int(counts.get("mowing", 0))
        not_mowing = int(counts.get("not_mowing", 0))
        decided = mowing + not_mowing
        summary = pd.DataFrame([{
            "reviewed_events": len(reviews),
            "confirmed_mowing": mowing,
            "confirmed_not_mowing": not_mowing,
            "uncertain": int(counts.get("uncertain", 0)),
            "precision_among_decided": mowing / decided if decided else np.nan,
        }])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    return summary
