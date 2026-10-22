from pathlib import Path

import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOSAIC = (
    PROJECT_ROOT
    / "dataset"
    / "georeferenced_landcover_base.tif"
)


def main():

    print("=" * 60)
    print("VERIFYING GEOREFERENCED LAND-COVER MOSAIC")
    print("=" * 60)

    with rasterio.open(MOSAIC) as src:

        print()
        print("File:")
        print(MOSAIC)

        print()
        print("Raster information:")
        print(f"Width:       {src.width}")
        print(f"Height:      {src.height}")
        print(f"Bands:       {src.count}")
        print(f"CRS:         {src.crs}")
        print(f"Resolution:  {src.res}")

        print()
        print("Bounds in EPSG:3857:")
        print(f"Left:   {src.bounds.left}")
        print(f"Bottom: {src.bounds.bottom}")
        print(f"Right:  {src.bounds.right}")
        print(f"Top:    {src.bounds.top}")

        # ----------------------------------------------------
        # Convert bounds to latitude / longitude
        # ----------------------------------------------------

        transformer = Transformer.from_crs(
            src.crs,
            "EPSG:4326",
            always_xy=True
        )

        left_lon, bottom_lat = transformer.transform(
            src.bounds.left,
            src.bounds.bottom
        )

        right_lon, top_lat = transformer.transform(
            src.bounds.right,
            src.bounds.top
        )

        print()
        print("Geographic bounds:")
        print(
            f"Longitude: "
            f"{left_lon:.6f} -> {right_lon:.6f}"
        )

        print(
            f"Latitude:  "
            f"{bottom_lat:.6f} -> {top_lat:.6f}"
        )

        # ----------------------------------------------------
        # Convert centre to UTM 44N
        # ----------------------------------------------------

        center_x = (
            src.bounds.left +
            src.bounds.right
        ) / 2

        center_y = (
            src.bounds.bottom +
            src.bounds.top
        ) / 2

        center_lon, center_lat = transformer.transform(
            center_x,
            center_y
        )

        utm_transformer = Transformer.from_crs(
            "EPSG:4326",
            "EPSG:32644",
            always_xy=True
        )

        utm_x, utm_y = utm_transformer.transform(
            center_lon,
            center_lat
        )

        print()
        print("Centre:")
        print(
            f"Latitude:  {center_lat:.6f}"
        )
        print(
            f"Longitude: {center_lon:.6f}"
        )

        print()
        print("Centre in EPSG:32644:")
        print(
            f"Easting:  {utm_x:.2f}"
        )
        print(
            f"Northing: {utm_y:.2f}"
        )

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()