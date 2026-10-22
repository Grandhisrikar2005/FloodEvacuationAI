
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLOOD_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_v2.tif"
)

PILOT_PATH = os.path.join(
    ROOT, "dataset", "zoom19_pilot",
    "vijayawada_zoom19_pilot.tif"
)

OUTPUT_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_aligned.tif"
)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with rasterio.open(PILOT_PATH) as pilot, \
         rasterio.open(FLOOD_PATH) as flood:

        # Create an output grid matching the pilot image exactly.
        aligned = np.full(
            (pilot.height, pilot.width),
            255,
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(flood, 1),
            destination=aligned,
            src_transform=flood.transform,
            src_crs=flood.crs,
            src_nodata=255,
            dst_transform=pilot.transform,
            dst_crs=pilot.crs,
            dst_nodata=255,
            resampling=Resampling.nearest
        )

        profile = pilot.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=255,
            compress="lzw"
        )

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            dst.write(aligned, 1)

        valid = aligned != 255
        candidates = aligned == 1

        print("\nALIGNED FLOOD CANDIDATE MASK")
        print("Output:", OUTPUT_PATH)
        print("Dimensions:", aligned.shape[1], "x", aligned.shape[0])
        print("CRS:", pilot.crs)
        print("Matches pilot transform:", True)
        print("Valid pixels:", int(valid.sum()))
        print("Candidate pixels:", int(candidates.sum()))
        print("NoData pixels:", int((~valid).sum()))

        if valid.any():
            print(
                "Candidate percentage of valid pixels:",
                round(100 * candidates.sum() / valid.sum(), 2), "%"
            )


if __name__ == "__main__":
    main()
