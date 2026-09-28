"""PlanetScope search, order, download, and cube orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from .aoi import create_square_aoi
from .config import WorkflowConfig
from .cube import build_planet_cube
from .io import read_json, write_json
from .utils import get_logger, stable_hash

LOGGER = get_logger(__name__)
DATA_SEARCH_URL = "https://api.planet.com/data/v1/quick-search"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"


def planet_session(api_key: str) -> requests.Session:
    """Create an authenticated Planet API session."""
    if not api_key:
        raise ValueError("Planet API key is required. Set PL_API_KEY.")
    session = requests.Session()
    session.auth = HTTPBasicAuth(api_key, "")
    return session


def search_planetscope(session: requests.Session, geometry: dict[str, Any], start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Search PlanetScope scenes intersecting an AOI."""
    payload = {
        "item_types": ["PSScene"],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": geometry},
                {"type": "DateRangeFilter", "field_name": "acquired", "config": {"gte": f"{start_date}T00:00:00.000Z", "lte": f"{end_date}T23:59:59.999Z"}},
                {"type": "PermissionFilter", "config": ["assets:download"]},
            ],
        },
    }
    features: list[dict[str, Any]] = []
    url: str | None = DATA_SEARCH_URL
    while url:
        response = session.post(url, json=payload) if url == DATA_SEARCH_URL else session.get(url)
        response.raise_for_status()
        data = response.json()
        features.extend(data.get("features", []))
        url = data.get("_links", {}).get("_next")
        payload = {}
    return sorted(features, key=lambda item: item["properties"]["acquired"])


def create_or_reuse_order(session: requests.Session, config: WorkflowConfig, features: list[dict[str, Any]], geometry: dict[str, Any]) -> dict[str, Any]:
    """Create or reuse a Planet order for the selected scenes."""
    item_ids = [feature["id"] for feature in features]
    signature = stable_hash([config.start_date, config.end_date, str(config.aoi_size), *item_ids])
    order_name = f"mowing-planetscope-{signature}"
    manifest_path = config.outputs_dir / "orders" / f"{order_name}.json"
    manifest = read_json(manifest_path)
    if manifest.get("id"):
        response = session.get(f"{ORDERS_URL}/{manifest['id']}")
        response.raise_for_status()
        LOGGER.info("Reusing Planet order %s", manifest["id"])
        return response.json()

    payload = {
        "name": order_name,
        "source_type": "scenes",
        "products": [{"item_ids": item_ids, "item_type": "PSScene", "product_bundle": "analytic_8b_sr_udm2"}],
        "tools": [{"clip": {"aoi": geometry}}],
    }
    response = session.post(ORDERS_URL, json=payload)
    response.raise_for_status()
    order = response.json()
    write_json(manifest_path, {"id": order["id"], "name": order_name, "item_ids": item_ids})
    LOGGER.info("Created Planet order %s", order["id"])
    return order


def download_order_results(session: requests.Session, order: dict[str, Any], output_dir: Path) -> list[Path]:
    """Download Planet order result files, skipping existing files."""
    order_id = order["id"]
    while order.get("state") != "success":
        if order.get("state") in {"failed", "partial", "cancelled"}:
            raise RuntimeError(f"Planet order ended with state {order.get('state')}")
        LOGGER.info("Waiting for Planet order %s, state=%s", order_id, order.get("state"))
        import time

        time.sleep(60)
        response = session.get(f"{ORDERS_URL}/{order_id}")
        response.raise_for_status()
        order = response.json()

    downloaded: list[Path] = []
    for result in order.get("_links", {}).get("results", []):
        location = result.get("location")
        if not location:
            continue
        target = output_dir / Path(result.get("name", Path(location).name)).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            downloaded.append(target)
            continue
        with session.get(location, stream=True) as response:
            response.raise_for_status()
            with target.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        downloaded.append(target)
    return downloaded


def download_planetscope(config: WorkflowConfig, api_key: str) -> Path:
    """Run the complete PlanetScope download workflow and build PlanetCube.nc."""
    config.create_directories()
    aoi = create_square_aoi(config.center_lat, config.center_lon, config.aoi_size, config.aoi_dir / f"aoi_{config.aoi_size}m.geojson")
    session = planet_session(api_key)
    search_manifest = config.outputs_dir / "search_results.json"
    cached = read_json(search_manifest)
    if cached.get("features"):
        features = cached["features"]
        LOGGER.info("Reusing cached search with %d scenes", len(features))
    else:
        features = search_planetscope(session, aoi["geometry"], config.start_date, config.end_date)
        write_json(search_manifest, {"features": features})
    order = create_or_reuse_order(session, config, features, aoi["geometry"])
    download_order_results(session, order, config.raw_dir)
    return build_planet_cube(config.raw_dir, config.planet_cube_path, features)
