from pathlib import Path

import rasterio
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MOSAIC = (
    PROJECT_ROOT
    / "dataset"
    / "georeferenced_landcover_utm44.tif"
)


def main():

    print("=" * 60)
    print("VERIFYING UTM LAND-COVER MOSAIC")
    print("=" * 60)

    print()
    print("File:")
    print(MOSAIC)

    with rasterio.open(MOSAIC) as src:

        print()
        print("Raster information:")
        print("Width:       ", src.width)
        print("Height:      ", src.height)
        print("Bands:       ", src.count)
        print("CRS:         ", src.crs)
        print("Resolution:  ", src.res)

        print()
        print("Bounds in EPSG:32644:")
        print("Left:   ", src.bounds.left)
        print("Bottom: ", src.bounds.bottom)
        print("Right:  ", src.bounds.right)
        print("Top:    ", src.bounds.top)

        # Convert bounds back to latitude/longitude
        transformer = Transformer.from_crs(
            "EPSG:32644",
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
        print(f"Longitude: {left_lon:.6f} -> {right_lon:.6f}")
        print(f"Latitude:  {bottom_lat:.6f} -> {top_lat:.6f}")

        # Centre
        center_x = (
            src.bounds.left + src.bounds.right
        ) / 2

        center_y = (
            src.bounds.bottom + src.bounds.top
        ) / 2

        center_lon, center_lat = transformer.transform(
            center_x,
            center_y
        )

        print()
        print("Centre:")
        print(f"Latitude:  {center_lat:.6f}")
        print(f"Longitude: {center_lon:.6f}")

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()