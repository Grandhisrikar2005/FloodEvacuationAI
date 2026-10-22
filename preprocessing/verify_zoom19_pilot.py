
from pathlib import Path

import rasterio
from rasterio.warp import transform


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MOSAIC_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "zoom19_pilot"
    / "vijayawada_zoom19_pilot.tif"
)

PROJECT_LAT = 16.5308984
PROJECT_LON = 80.6176758


def main():
    if not MOSAIC_PATH.exists():
        print(f"ERROR: Mosaic not found:\n{MOSAIC_PATH}")
        return

    with rasterio.open(MOSAIC_PATH) as src:
        print("=" * 60)
        print("ZOOM-19 PILOT MOSAIC VERIFICATION")
        print("=" * 60)

        print(f"File       : {MOSAIC_PATH.name}")
        print(f"Dimensions : {src.width} x {src.height}")
        print(f"Bands      : {src.count}")
        print(f"Data type  : {src.dtypes}")
        print(f"CRS        : {src.crs}")
        print(f"Transform  : {src.transform}")
        print(f"Bounds     : {src.bounds}")
        print(f"Pixel size : {src.res}")

        left, bottom, right, top = src.bounds

        lon_values, lat_values = transform(
            src.crs,
            "EPSG:4326",
            [left, right],
            [bottom, top],
        )

        print("\nGEOGRAPHIC COVERAGE")
        print(
            f"Longitude: {min(lon_values):.6f} "
            f"to {max(lon_values):.6f}"
        )
        print(
            f"Latitude : {min(lat_values):.6f} "
            f"to {max(lat_values):.6f}"
        )

        x_values, y_values = transform(
            "EPSG:4326",
            src.crs,
            [PROJECT_LON],
            [PROJECT_LAT],
        )

        x, y = x_values[0], y_values[0]
        inside = (
            left <= x <= right
            and bottom <= y <= top
        )

        print("\nPROJECT CENTRE CHECK")
        print(f"Project coordinate inside mosaic: {inside}")

        if inside:
            row, col = src.index(x, y)
            print(f"Pixel position: row={row}, column={col}")

            sample = src.read(
                indexes=[1, 2, 3],
                window=((row, row + 1), (col, col + 1)),
            )
            print(f"Centre pixel RGB: {sample[:, 0, 0].tolist()}")

        print("\nDATA VALIDITY")
        print(f"File size: {MOSAIC_PATH.stat().st_size / (1024**2):.2f} MB")
        print(f"No-data value: {src.nodata}")

        print("\nVerification complete.")


if __name__ == "__main__":
    main()
