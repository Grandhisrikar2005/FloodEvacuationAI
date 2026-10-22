
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.join(
    ROOT, "dataset", "temporal", "krishna_flood_2024",
    "flood_analysis"
)

CHANGE_PATH = os.path.join(
    BASE, "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

MASK_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_aligned.tif"
)


def main():
    with rasterio.open(MASK_PATH) as mask_src, \
         rasterio.open(CHANGE_PATH) as change_src:

        mask = mask_src.read(1)
        aligned_change = np.full(
            mask.shape, np.nan, dtype=np.float32
        )

        reproject(
            source=rasterio.band(change_src, 1),
            destination=aligned_change,
            src_transform=change_src.transform,
            src_crs=change_src.crs,
            src_nodata=change_src.nodata,
            dst_transform=mask_src.transform,
            dst_crs=mask_src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest
        )

    candidates = mask == 1
    values = aligned_change[candidates]
    values = values[np.isfinite(values)]

    print("\nSAR CHANGE CANDIDATE INSPECTION")
    print("Candidate pixels in mask:", int(candidates.sum()))
    print("Candidates with valid SAR values:", len(values))

    if len(values):
        print("Minimum change (dB):", float(values.min()))
        print("Maximum change (dB):", float(values.max()))
        print("Mean change (dB):", round(float(values.mean()), 3))
        print("Median change (dB):", round(float(np.median(values)), 3))
        print("At or below -3 dB:", int((values <= -3).sum()))

    print("\nThese statistics diagnose the threshold; they do not prove flooding.")


if __name__ == "__main__":
    main()
