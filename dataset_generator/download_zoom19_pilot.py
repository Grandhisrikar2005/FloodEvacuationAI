
import math
import time
from io import BytesIO
from pathlib import Path

import requests
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin


# --------------------------------------------------
# PROJECT SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "zoom19_pilot"
TILES_DIR = OUTPUT_DIR / "tiles"
MOSAIC_PATH = OUTPUT_DIR / "vijayawada_zoom19_pilot.tif"

CENTER_LAT = 16.5308984
CENTER_LON = 80.6176758

ZOOM = 19
TILE_SIZE = 256

# 9 x 9 tiles = 81 tiles
GRID_RADIUS = 4

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def lonlat_to_tile(lon, lat, zoom):
    """Convert longitude/latitude to XYZ tile coordinates."""
    lat = max(min(lat, 85.05112878), -85.05112878)

    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)

    lat_rad = math.radians(lat)
    y = int(
        (1.0 - math.asinh(math.tan(lat_rad)) / math.pi)
        / 2.0 * n
    )

    return x, y


def download_tile(session, x, y):
    """Download one XYZ tile, retrying transient failures."""
    tile_path = TILES_DIR / f"z{ZOOM}_x{x}_y{y}.jpg"

    if tile_path.exists() and tile_path.stat().st_size > 1000:
        try:
            with Image.open(tile_path) as image:
                image.verify()
            return tile_path
        except Exception:
            tile_path.unlink(missing_ok=True)

    url = TILE_URL.format(z=ZOOM, x=x, y=y)

    for attempt in range(1, 5):
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as image:
                image = image.convert("RGB")
                if image.size != (TILE_SIZE, TILE_SIZE):
                    raise ValueError(
                        f"Unexpected tile size: {image.size}"
                    )
                image.save(tile_path, format="JPEG", quality=95)

            return tile_path

        except Exception as error:
            print(
                f"Retry {attempt}/4 for tile ({x}, {y}): {error}"
            )
            if attempt < 4:
                time.sleep(attempt * 2)

    raise RuntimeError(f"Could not download tile ({x}, {y})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    center_x, center_y = lonlat_to_tile(
        CENTER_LON, CENTER_LAT, ZOOM
    )

    min_x = center_x - GRID_RADIUS
    max_x = center_x + GRID_RADIUS
    min_y = center_y - GRID_RADIUS
    max_y = center_y + GRID_RADIUS

    columns = max_x - min_x + 1
    rows = max_y - min_y + 1

    print("=" * 60)
    print("ZOOM-19 PILOT IMAGERY DOWNLOAD")
    print("=" * 60)
    print(f"Centre coordinate : {CENTER_LAT}, {CENTER_LON}")
    print(f"Zoom              : {ZOOM}")
    print(f"Centre tile       : X={center_x}, Y={center_y}")
    print(f"Grid              : {columns} columns x {rows} rows")
    print(f"Output directory  : {OUTPUT_DIR}")
    print("\nDownloading tiles...")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "FloodEvacuationAI/1.0"
    })

    mosaic = np.zeros(
        (rows * TILE_SIZE, columns * TILE_SIZE, 3),
        dtype=np.uint8,
    )

    for row_index, y in enumerate(range(min_y, max_y + 1)):
        for col_index, x in enumerate(range(min_x, max_x + 1)):
            tile_path = download_tile(session, x, y)

            with Image.open(tile_path) as image:
                tile = np.asarray(image.convert("RGB"))

            row_start = row_index * TILE_SIZE
            col_start = col_index * TILE_SIZE

            mosaic[
                row_start:row_start + TILE_SIZE,
                col_start:col_start + TILE_SIZE,
            ] = tile

            completed = row_index * columns + col_index + 1
            print(
                f"[{completed:02d}/{rows * columns}] "
                f"Downloaded tile X={x}, Y={y}"
            )

    # Web Mercator XYZ tile georeferencing
    world_width = 2 * math.pi * 6378137.0
    tile_span = world_width / (2 ** ZOOM)
    pixel_size = tile_span / TILE_SIZE

    left = -math.pi * 6378137.0 + min_x * tile_span
    top = math.pi * 6378137.0 - min_y * tile_span

    transform = from_origin(left, top, pixel_size, pixel_size)

    with rasterio.open(
        MOSAIC_PATH,
        "w",
        driver="GTiff",
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        for band in range(3):
            dst.write(mosaic[:, :, band], band + 1)

        dst.update_tags(
            source="ArcGIS World Imagery",
            zoom=str(ZOOM),
            attribution=(
                "Imagery source: Esri, Maxar, Earthstar Geographics "
                "and other contributors, as applicable to the service."
            ),
        )

    print("\n" + "=" * 60)
    print("PILOT MOSAIC CREATED")
    print("=" * 60)
    print(f"Tiles downloaded : {rows * columns}")
    print(f"Image dimensions : {mosaic.shape[1]} x {mosaic.shape[0]}")
    print(f"CRS              : EPSG:3857")
    print(f"Pixel size       : {pixel_size:.4f} projected metres")
    print(f"Saved to         : {MOSAIC_PATH}")
    print("\nNext: verify the mosaic before classifying it.")


if __name__ == "__main__":
    main()
