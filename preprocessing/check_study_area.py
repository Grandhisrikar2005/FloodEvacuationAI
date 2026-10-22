
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
BASE = DATASET / "temporal" / "krishna_flood_2024"

FILES = {
    "UTM land-cover base": DATASET / "georeferenced_landcover_utm44.tif",
    "Flood reference": (
        BASE / "reference" /
        "NRSC_APSAC_Vijayawada_reference_georef.tif"
    ),
    "SAR candidate mask": (
        BASE / "flood_analysis" /
        "best_flood_mask_v3_2024-09-01.tif"
    ),
}


def main():
    with rasterio.open(FILES["UTM land-cover base"]) as land:
        print("LAND-COVER BASE")
        print("CRS:", land.crs)
        print("Bounds:", land.bounds)
        print("Dimensions:", land.width, "x", land.height)

        rgb = land.read()
        pixels = np.moveaxis(rgb, 0, -1).reshape(-1, 3)

        unique, counts = np.unique(
            pixels, axis=0, return_counts=True
        )
        order = np.argsort(counts)[::-1]

        print("\nMost common RGB colors:")
        for i in order[:10]:
            print(tuple(unique[i]), "pixels:", int(counts[i]))

        land_bounds = land.bounds

    print("\nOVERLAP WITH STUDY RASTERS")

    for name in ("Flood reference", "SAR candidate mask"):
        with rasterio.open(FILES[name]) as src:
            bounds = transform_bounds(
                land.crs, src.crs, *land_bounds, densify_pts=21
            )

            left = max(bounds[0], src.bounds.left)
            bottom = max(bounds[1], src.bounds.bottom)
            right = min(bounds[2], src.bounds.right)
            top = min(bounds[3], src.bounds.top)

            overlaps = left < right and bottom < top

            print(f"\n{name}")
            print("CRS:", src.crs)
            print("Overlap:", overlaps)

            if overlaps:
                print(
                    "Overlap bounds:",
                    (left, bottom, right, top)
                )
                print(
                    "Approximate overlap area:",
                    round((right - left) * (top - bottom) / 1e6, 3),
                    "square km (in the target CRS)"
                )


if __name__ == "__main__":
    main()
