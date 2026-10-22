
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_PATH = os.path.join(
    ROOT, "dataset", "temporal", "krishna_flood_2024",
    "reference", "reference_flood_mask_2024-09-01.tif"
)

CANDIDATE_PATH = os.path.join(
    ROOT, "outputs", "landcover",
    "vijayawada_flood_candidates_aligned.tif"
)


def main():
    with rasterio.open(CANDIDATE_PATH) as candidate_src, \
         rasterio.open(REFERENCE_PATH) as ref_src:

        candidate = candidate_src.read(1)
        reference = np.zeros(candidate.shape, dtype=np.uint8)

        reference_valid = np.zeros(candidate.shape, dtype=np.uint8)

        # Reproject the reference labels using nearest-neighbor resampling.
        reproject(
            source=rasterio.band(ref_src, 1),
            destination=reference,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=ref_src.nodata,
            dst_transform=candidate_src.transform,
            dst_crs=candidate_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        # Reproject a validity layer separately so reference NoData
        # does not get treated as a non-flood label.
        ref_valid_source = np.ones(
            (ref_src.height, ref_src.width), dtype=np.uint8
        )
        if ref_src.nodata is not None:
            raw = ref_src.read(1)
            ref_valid_source[raw == ref_src.nodata] = 0

        reproject(
            source=ref_valid_source,
            destination=reference_valid,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=0,
            dst_transform=candidate_src.transform,
            dst_crs=candidate_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

    # Candidate NoData and reference NoData are excluded.
    valid = (candidate != 255) & (reference_valid == 1)
    pred = (candidate == 1) & valid
    truth = (reference == 1) & valid

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth & valid))
    fn = int(np.sum(~pred & truth & valid))
    tn = int(np.sum(~pred & ~truth & valid))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0

    print("\nVIJAYAWADA FLOOD MASK VALIDATION")
    print("Valid comparison pixels:", int(valid.sum()))
    print("True positives:", tp)
    print("False positives:", fp)
    print("False negatives:", fn)
    print("True negatives:", tn)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")
    print("\nInterpret results cautiously: reference NoData is excluded.")


if __name__ == "__main__":
    main()
