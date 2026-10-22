from pathlib import Path

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = PROJECT_ROOT / "dataset" / "georeferenced_landcover_base.tif"

OUTPUT = (
    PROJECT_ROOT
    / "dataset"
    / "georeferenced_landcover_utm44.tif"
)


def main():

    print("=" * 60)
    print("REPROJECTING LAND-COVER MOSAIC")
    print("=" * 60)

    print()
    print("Input:")
    print(INPUT)

    print()
    print("Output:")
    print(OUTPUT)

    target_crs = "EPSG:32644"

    with rasterio.open(INPUT) as src:

        print()
        print("Source CRS:", src.crs)
        print("Source size:", src.width, "x", src.height)

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds
        )

        profile = src.profile.copy()

        profile.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "driver": "GTiff",
                "compress": "lzw"
            }
        )

        with rasterio.open(OUTPUT, "w", **profile) as dst:

            for band in range(1, src.count + 1):

                print(f"Reprojecting band {band}/{src.count}...")

                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

    print()
    print("=" * 60)
    print("REPROJECTION COMPLETE")
    print("=" * 60)

    with rasterio.open(OUTPUT) as src:

        print()
        print("Final raster:")
        print("Width:       ", src.width)
        print("Height:      ", src.height)
        print("Bands:       ", src.count)
        print("CRS:         ", src.crs)
        print("Resolution:  ", src.res)
        print("Bounds:")
        print("  Left:      ", src.bounds.left)
        print("  Bottom:    ", src.bounds.bottom)
        print("  Right:     ", src.bounds.right)
        print("  Top:       ", src.bounds.top)

    print()
    print("Saved successfully:")
    print(OUTPUT)


if __name__ == "__main__":
    main()