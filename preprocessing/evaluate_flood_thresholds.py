import csv
import os

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage


# ============================================================
# FILE PATHS
# ============================================================

CHANGE_MAP = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\flood_analysis"
    r"\sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

REFERENCE_MASK = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\reference_flood_mask_2024-09-01.tif"
)

OUTPUT_CSV = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\flood_analysis"
    r"\threshold_component_analysis.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

THRESHOLDS = [
    -2.0,
    -3.0,
    -4.0,
    -5.0
]

MIN_REGION_SIZES = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000
]

# 8-connectivity
STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8
)

# Extra area around the APSAC reference region
MARGIN_PIXELS = 300


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("SENTINEL-1 THRESHOLD + CONNECTED-COMPONENT ANALYSIS")
print("=" * 80)

print("\nThresholds:")
print(THRESHOLDS)

print("\nMinimum region sizes:")
print(MIN_REGION_SIZES)


# ============================================================
# LOAD SAR CHANGE MAP
# ============================================================

print("\nLoading Sentinel-1 change map...")

with rasterio.open(CHANGE_MAP) as sar_src:

    sar_change = sar_src.read(1)

    sar_transform = sar_src.transform
    sar_crs = sar_src.crs

    sar_width = sar_src.width
    sar_height = sar_src.height

    sar_nodata = sar_src.nodata

    print("\nSAR:")
    print(f"Dimensions: {sar_width} x {sar_height}")
    print(f"CRS: {sar_crs}")
    print(f"Resolution: {sar_src.res}")
    print(f"Nodata: {sar_nodata}")


# ============================================================
# LOAD APSAC REFERENCE
# ============================================================

print("\nLoading APSAC reference mask...")

with rasterio.open(REFERENCE_MASK) as ref_src:

    reference_mask = ref_src.read(1)

    ref_transform = ref_src.transform
    ref_crs = ref_src.crs

    ref_width = ref_src.width
    ref_height = ref_src.height

    print("\nAPSAC:")
    print(f"Dimensions: {ref_width} x {ref_height}")
    print(f"CRS: {ref_crs}")
    print(f"Resolution: {ref_src.res}")


# ============================================================
# SAR VALID MASK
# ============================================================

if sar_nodata is not None:

    valid_sar = (
        np.isfinite(sar_change)
        & (sar_change != sar_nodata)
    )

else:

    valid_sar = np.isfinite(sar_change)


print(
    "\nValid SAR pixels:",
    int(np.sum(valid_sar))
)


# ============================================================
# REPROJECT APSAC TO SAR GRID
# ============================================================

print("\nReprojecting APSAC reference to SAR grid...")
print("Please wait...")


reference_on_sar = np.zeros(
    (sar_height, sar_width),
    dtype=np.uint8
)

reference_coverage = np.ones(
    (ref_height, ref_width),
    dtype=np.uint8
)

reference_valid_on_sar = np.zeros(
    (sar_height, sar_width),
    dtype=np.uint8
)


reproject(
    source=reference_mask.astype(np.uint8),
    destination=reference_on_sar,

    src_transform=ref_transform,
    src_crs=ref_crs,

    dst_transform=sar_transform,
    dst_crs=sar_crs,

    resampling=Resampling.nearest
)


reproject(
    source=reference_coverage,
    destination=reference_valid_on_sar,

    src_transform=ref_transform,
    src_crs=ref_crs,

    dst_transform=sar_transform,
    dst_crs=sar_crs,

    resampling=Resampling.nearest
)


reference_on_sar = (
    reference_on_sar > 0
)

reference_valid_on_sar = (
    reference_valid_on_sar > 0
)


# ============================================================
# FIND APSAC REGION
# ============================================================

rows, cols = np.where(
    reference_valid_on_sar
)

if len(rows) == 0:
    raise RuntimeError(
        "Could not locate APSAC reference area on SAR grid."
    )


min_row = max(
    0,
    int(rows.min()) - MARGIN_PIXELS
)

max_row = min(
    sar_height,
    int(rows.max()) + MARGIN_PIXELS + 1
)

min_col = max(
    0,
    int(cols.min()) - MARGIN_PIXELS
)

max_col = min(
    sar_width,
    int(cols.max()) + MARGIN_PIXELS + 1
)


print("\nEvaluation crop:")
print(
    f"Rows: {min_row} -> {max_row}"
)

print(
    f"Columns: {min_col} -> {max_col}"
)


# ============================================================
# CROP SAR + REFERENCE
# ============================================================

sar_crop = sar_change[
    min_row:max_row,
    min_col:max_col
]

valid_sar_crop = valid_sar[
    min_row:max_row,
    min_col:max_col
]

reference_crop = reference_on_sar[
    min_row:max_row,
    min_col:max_col
]

reference_valid_crop = reference_valid_on_sar[
    min_row:max_row,
    min_col:max_col
]


# ============================================================
# COMMON EVALUATION AREA
# ============================================================

evaluation_mask = (
    valid_sar_crop
    & reference_valid_crop
)

reference_flood = (
    reference_crop
    & evaluation_mask
)


evaluation_pixels = int(
    np.sum(evaluation_mask)
)

reference_flood_pixels = int(
    np.sum(reference_flood)
)


print("\nEvaluation pixels:")
print(
    f"{evaluation_pixels:,}"
)

print(
    "Reference flood pixels:",
    f"{reference_flood_pixels:,}"
)


# ============================================================
# PIXEL AREA
# ============================================================

pixel_width = abs(
    sar_transform.a
)

pixel_height = abs(
    sar_transform.e
)

pixel_area_km2 = (
    pixel_width
    * pixel_height
    / 1_000_000
)


# ============================================================
# FUNCTION: CALCULATE METRICS
# ============================================================

def calculate_metrics(
    predicted_flood,
    reference_flood,
    evaluation_mask
):

    tp = int(
        np.sum(
            predicted_flood
            & reference_flood
        )
    )

    fp = int(
        np.sum(
            predicted_flood
            & (~reference_flood)
            & evaluation_mask
        )
    )

    fn = int(
        np.sum(
            (~predicted_flood)
            & reference_flood
        )
    )

    tn = int(
        np.sum(
            (~predicted_flood)
            & (~reference_flood)
            & evaluation_mask
        )
    )

    # Precision
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # Recall
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # F1
    if precision + recall > 0:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    # IoU
    union = (
        tp
        + fp
        + fn
    )

    if union > 0:
        iou = tp / union
    else:
        iou = 0.0

    # Accuracy
    total = (
        tp
        + fp
        + fn
        + tn
    )

    if total > 0:
        accuracy = (
            tp + tn
        ) / total
    else:
        accuracy = 0.0

    flood_pixels = int(
        np.sum(predicted_flood)
    )

    flood_area_km2 = (
        flood_pixels
        * pixel_area_km2
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "flood_pixels": flood_pixels,
        "flood_area_km2": flood_area_km2
    }


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TEST THRESHOLDS
# ============================================================

print("\n")
print("=" * 80)
print("RUNNING THRESHOLD + REGION FILTERING")
print("=" * 80)


for threshold in THRESHOLDS:

    print("\n")
    print("-" * 80)
    print(
        f"THRESHOLD = {threshold} dB"
    )
    print("-" * 80)

    # --------------------------------------------------------
    # Raw threshold mask
    # --------------------------------------------------------

    raw_flood = (
        valid_sar_crop
        & (sar_crop <= threshold)
    )

    raw_pixels = int(
        np.sum(raw_flood)
    )

    print(
        f"Raw flood pixels: {raw_pixels:,}"
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    print(
        "Finding connected regions..."
    )

    labels, number_of_regions = ndimage.label(
        raw_flood,
        structure=STRUCTURE
    )

    print(
        f"Connected regions found: "
        f"{number_of_regions:,}"
    )

    # --------------------------------------------------------
    # Region sizes
    # --------------------------------------------------------

    region_sizes = np.bincount(
        labels.ravel()
    )

    # Ignore background label 0
    region_sizes[0] = 0

    # --------------------------------------------------------
    # Test minimum region sizes
    # --------------------------------------------------------

    for min_size in MIN_REGION_SIZES:

        print(
            f"\n  Minimum region size: "
            f"{min_size} pixels"
        )

        # Keep only sufficiently large regions
        keep_regions = (
            region_sizes >= min_size
        )

        filtered_flood = (
            keep_regions[labels]
            & evaluation_mask
        )

        metrics = calculate_metrics(
            filtered_flood,
            reference_flood,
            evaluation_mask
        )

        result = {

            "threshold_db":
                threshold,

            "min_region_pixels":
                min_size,

            "TP":
                metrics["TP"],

            "FP":
                metrics["FP"],

            "FN":
                metrics["FN"],

            "TN":
                metrics["TN"],

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "iou":
                metrics["iou"],

            "flood_pixels":
                metrics["flood_pixels"],

            "flood_area_km2":
                metrics["flood_area_km2"]
        }

        results.append(result)

        print(
            f"    Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"    Recall:    "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"    F1:        "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"    IoU:       "
            f"{metrics['iou']:.4f}"
        )

        print(
            f"    Area:      "
            f"{metrics['flood_area_km2']:.2f} km²"
        )


# ============================================================
# FIND BEST RESULTS
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x["f1"]
)

best_iou = max(
    results,
    key=lambda x: x["iou"]
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("BEST FILTERED RESULTS")
print("=" * 80)


print("\nBEST F1:")
print(
    f"Threshold: "
    f"{best_f1['threshold_db']} dB"
)

print(
    f"Minimum region: "
    f"{best_f1['min_region_pixels']} pixels"
)

print(
    f"Precision: "
    f"{best_f1['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_f1['recall']:.4f}"
)

print(
    f"F1: "
    f"{best_f1['f1']:.4f}"
)

print(
    f"IoU: "
    f"{best_f1['iou']:.4f}"
)

print(
    f"Flood area: "
    f"{best_f1['flood_area_km2']:.2f} km²"
)


print("\nBEST IoU:")
print(
    f"Threshold: "
    f"{best_iou['threshold_db']} dB"
)

print(
    f"Minimum region: "
    f"{best_iou['min_region_pixels']} pixels"
)

print(
    f"Precision: "
    f"{best_iou['precision']:.4f}"
)

print(
    f"Recall: "
    f"{best_iou['recall']:.4f}"
)

print(
    f"F1: "
    f"{best_iou['f1']:.4f}"
)

print(
    f"IoU: "
    f"{best_iou['iou']:.4f}"
)

print(
    f"Flood area: "
    f"{best_iou['flood_area_km2']:.2f} km²"
)


# ============================================================
# COMPACT TABLE
# ============================================================

print("\n")
print("=" * 100)
print("ALL FILTERING RESULTS")
print("=" * 100)

print(
    f"{'Threshold':>10} "
    f"{'MinPix':>10} "
    f"{'Precision':>12} "
    f"{'Recall':>12} "
    f"{'F1':>12} "
    f"{'IoU':>12} "
    f"{'Area km²':>14}"
)

print("-" * 100)


for result in results:

    print(
        f"{result['threshold_db']:>10.1f} "
        f"{result['min_region_pixels']:>10} "
        f"{result['precision']:>12.4f} "
        f"{result['recall']:>12.4f} "
        f"{result['f1']:>12.4f} "
        f"{result['iou']:>12.4f} "
        f"{result['flood_area_km2']:>14.2f}"
    )


# ============================================================
# SAVE CSV
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)

fieldnames = list(
    results[0].keys()
)


with open(
    OUTPUT_CSV,
    "w",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)

print(
    "Results saved to:"
)

print(
    OUTPUT_CSV
)

print("=" * 80)

print(
    "\nTHRESHOLD + CONNECTED-COMPONENT ANALYSIS COMPLETE"
)