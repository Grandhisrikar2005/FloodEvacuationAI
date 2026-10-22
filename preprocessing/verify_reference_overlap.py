
import os
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE = os.path.join(
    ROOT, "dataset", "temporal", "krishna_flood_2024",
    "reference", "reference_flood_mask_2024-09-01.tif"
)

PILOT = os.path.join(
    ROOT, "dataset", "zoom19_pilot",
    "vijayawada_zoom19_pilot.tif"
)


def main():
    print("REFERENCE MASK FULL-RASTER CHECK")

    counts = {}

    with rasterio.open(REFERENCE) as src:
        print("Reference CRS:", src.crs)
        print("Reference bounds:", src.bounds)
        print("Reference NoData:", src.nodata)

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window)
            values, numbers = np.unique(arr, return_counts=True)

            for value, number in zip(values, numbers):
                counts[int(value)] = counts.get(int(value), 0) + int(number)

        print("\nPixel counts across entire mask:")
        for value, number in sorted(counts.items()):
            print(f"  Value {value}: {number:,}")

        if src.nodata is not None:
            nodata_count = counts.get(int(src.nodata), 0)
            print("NoData pixel count:", f"{nodata_count:,}")

    print("\nPILOT / REFERENCE OVERLAP CHECK")

    with rasterio.open(PILOT) as pilot, rasterio.open(REFERENCE) as ref:
        pilot_bounds_in_ref = transform_bounds(
            pilot.crs,
            ref.crs,
            *pilot.bounds,
            densify_pts=21
        )

        print("Pilot CRS:", pilot.crs)
        print("Pilot bounds transformed to reference CRS:")
        print(pilot_bounds_in_ref)

        left = max(pilot_bounds_in_ref[0], ref.bounds.left)
        bottom = max(pilot_bounds_in_ref[1], ref.bounds.bottom)
        right = min(pilot_bounds_in_ref[2], ref.bounds.right)
        top = min(pilot_bounds_in_ref[3], ref.bounds.top)

        if left < right and bottom < top:
            print("Result: BOUNDS OVERLAP")
            print("Overlap bounds:", (left, bottom, right, top))
        else:
            print("Result: NO BOUNDS OVERLAP")
            print("The pilot and reference do not cover the same area.")


if __name__ == "__main__":
    main()
