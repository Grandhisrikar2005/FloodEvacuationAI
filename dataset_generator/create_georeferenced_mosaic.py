from pathlib import Path
import math

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_FILE = (
    DATASET_DIR
    / "georeferenced_landcover_base.tif"
)


# ============================================================
# TILE SETTINGS
# ============================================================

ZOOM = 15

TILE_SIZE = 256

# Your existing grid
MIN_TILE_X = 23722
MAX_TILE_X = 23726

MIN_TILE_Y = 14858
MAX_TILE_Y = 14862

NUM_X = MAX_TILE_X - MIN_TILE_X + 1
NUM_Y = MAX_TILE_Y - MIN_TILE_Y + 1


# ============================================================
# WEB MERCATOR FUNCTIONS
# ============================================================

def tile_to_webmercator_bounds(
    tile_x,
    tile_y,
    zoom
):
    """
    Return tile bounds in EPSG:3857.

    Returns:
        left, bottom, right, top
    """

    world_size = 40075016.68557849

    resolution = (
        world_size /
        (2 ** zoom) /
        TILE_SIZE
    )

    left = (
        tile_x *
        TILE_SIZE *
        resolution
        -
        world_size / 2
    )

    right = (
        (tile_x + 1) *
        TILE_SIZE *
        resolution
        -
        world_size / 2
    )

    top = (
        world_size / 2
        -
        tile_y *
        TILE_SIZE *
        resolution
    )

    bottom = (
        world_size / 2
        -
        (tile_y + 1) *
        TILE_SIZE *
        resolution
    )

    return left, bottom, right, top


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CREATING GEOREFERENCED LAND-COVER BASE MOSAIC")
    print("=" * 60)

    print()
    print(f"Zoom: {ZOOM}")
    print(
        f"Grid: "
        f"{NUM_X} x {NUM_Y}"
    )

    print(
        f"Total tiles: "
        f"{NUM_X * NUM_Y}"
    )

    # --------------------------------------------------------
    # Check all tiles
    # --------------------------------------------------------

    print()
    print("Checking tiles...")

    missing = []

    for tile_y in range(
        MIN_TILE_Y,
        MAX_TILE_Y + 1
    ):

        for tile_x in range(
            MIN_TILE_X,
            MAX_TILE_X + 1
        ):

            tile_number = (
                (tile_y - MIN_TILE_Y) *
                NUM_X +
                (tile_x - MIN_TILE_X) +
                1
            )

            tile_path = (
                DATASET_DIR /
                f"tile_{tile_number}.jpg"
            )

            if not tile_path.exists():
                missing.append(
                    str(tile_path)
                )

    if missing:

        print()
        print("ERROR: Missing tiles:")

        for path in missing:
            print(path)

        return

    print("All 25 tiles found.")

    # --------------------------------------------------------
    # Calculate mosaic dimensions
    # --------------------------------------------------------

    mosaic_width = (
        NUM_X *
        TILE_SIZE
    )

    mosaic_height = (
        NUM_Y *
        TILE_SIZE
    )

    # --------------------------------------------------------
    # Get Web Mercator bounds
    # --------------------------------------------------------

    left, bottom, _, top = (
        tile_to_webmercator_bounds(
            MIN_TILE_X,
            MAX_TILE_Y,
            ZOOM
        )
    )

    _, _, right, _ = (
        tile_to_webmercator_bounds(
            MAX_TILE_X,
            MIN_TILE_Y,
            ZOOM
        )
    )

    # Pixel resolution
    world_size = 40075016.68557849

    pixel_size = (
        world_size /
        (2 ** ZOOM) /
        TILE_SIZE
    )

    transform = from_origin(
        left,
        top,
        pixel_size,
        pixel_size
    )

    print()
    print("Mosaic information:")
    print(
        f"Dimensions: "
        f"{mosaic_width} x "
        f"{mosaic_height}"
    )

    print(
        f"Pixel size: "
        f"{pixel_size:.6f} m"
    )

    print(
        f"Bounds:"
    )

    print(
        f"  Left:   {left:.3f}"
    )

    print(
        f"  Bottom: {bottom:.3f}"
    )

    print(
        f"  Right:  {right:.3f}"
    )

    print(
        f"  Top:    {top:.3f}"
    )

    print()
    print("CRS: EPSG:3857")

    # --------------------------------------------------------
    # Create mosaic
    # --------------------------------------------------------

    print()
    print("Creating RGB mosaic...")

    mosaic = np.zeros(
        (
            3,
            mosaic_height,
            mosaic_width
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Place each tile
    # --------------------------------------------------------

    for row, tile_y in enumerate(
        range(
            MIN_TILE_Y,
            MAX_TILE_Y + 1
        )
    ):

        for col, tile_x in enumerate(
            range(
                MIN_TILE_X,
                MAX_TILE_X + 1
            )
        ):

            tile_number = (
                row *
                NUM_X +
                col +
                1
            )

            tile_path = (
                DATASET_DIR /
                f"tile_{tile_number}.jpg"
            )

            print(
                f"Adding tile_{tile_number}.jpg "
                f"({row + 1},{col + 1})"
            )

            image = Image.open(
                tile_path
            ).convert("RGB")

            image_array = np.array(
                image
            )

            if image_array.shape != (
                TILE_SIZE,
                TILE_SIZE,
                3
            ):

                print(
                    "ERROR: Unexpected tile size:"
                )

                print(
                    image_array.shape
                )

                return

            row_start = (
                row *
                TILE_SIZE
            )

            row_end = (
                row_start +
                TILE_SIZE
            )

            col_start = (
                col *
                TILE_SIZE
            )

            col_end = (
                col_start +
                TILE_SIZE
            )

            # RGB → rasterio band order
            mosaic[
                :,
                row_start:row_end,
                col_start:col_end
            ] = np.transpose(
                image_array,
                (2, 0, 1)
            )

    # --------------------------------------------------------
    # Save GeoTIFF
    # --------------------------------------------------------

    print()
    print("Saving GeoTIFF...")

    profile = {
        "driver": "GTiff",
        "height": mosaic_height,
        "width": mosaic_width,
        "count": 3,
        "dtype": rasterio.uint8,
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "lzw"
    }

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        dst.write(mosaic)

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print(
        f"Size: "
        f"{mosaic_width} x "
        f"{mosaic_height}"
    )

    print(
        f"CRS: EPSG:3857"
    )

    print(
        f"Resolution: "
        f"{pixel_size:.6f} m"
    )


if __name__ == "__main__":
    main()