from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from scipy import ndimage
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from matplotlib.patches import Patch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHANGE_MAP = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

REFERENCE_MASK = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "reference"
    / "reference_flood_mask_2024-09-01.tif"
)

OUTPUT_MASK = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "validated_flood_mask_2024-09-01.tif"
)

OUTPUT_IMAGE = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
    / "validated_flood_mask_vs_apsac.png"
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD_DB = -2.0
MIN_COMPONENT_PIXELS = 1000

# Used ONLY while performing connected-component filtering
MARGIN_PIXELS = 300

STRUCTURE = np.ones(
    (3, 3),
    dtype=np.uint8
)

MAX_DISPLAY_SIZE = 1400


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VALIDATED FLOOD MASK - CORRECTED EVALUATION")
    print("=" * 60)

    print()
    print(f"Threshold: {THRESHOLD_DB} dB")
    print(f"Minimum component: {MIN_COMPONENT_PIXELS} pixels")
    print(f"Processing margin: {MARGIN_PIXELS} pixels")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not CHANGE_MAP.exists():
        print()
        print("ERROR: SAR change map not found:")
        print(CHANGE_MAP)
        return

    if not REFERENCE_MASK.exists():
        print()
        print("ERROR: APSAC reference mask not found:")
        print(REFERENCE_MASK)
        return

    OUTPUT_MASK.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # OPEN INPUTS
    # ========================================================

    with rasterio.open(CHANGE_MAP) as sar_src, \
         rasterio.open(REFERENCE_MASK) as ref_src:

        print()
        print("SAR:")
        print(f"  Size: {sar_src.width} x {sar_src.height}")
        print(f"  CRS: {sar_src.crs}")
        print(f"  Resolution: {sar_src.res}")

        print()
        print("APSAC:")
        print(f"  Size: {ref_src.width} x {ref_src.height}")
        print(f"  CRS: {ref_src.crs}")
        print(f"  Resolution: {ref_src.res}")

        # ====================================================
        # EXACT APSAC FOOTPRINT
        # ====================================================

        ref_bounds = ref_src.bounds

        print()
        print("APSAC footprint:")
        print(ref_bounds)

        # ----------------------------------------------------
        # Exact evaluation window
        #
        # This is used for final metrics.
        # NO margin is included here.
        # ----------------------------------------------------

        evaluation_window = from_bounds(
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top,
            transform=sar_src.transform
        )

        evaluation_window = (
            evaluation_window
            .round_offsets()
            .round_lengths()
        )

        eval_col_start = max(
            0,
            int(evaluation_window.col_off)
        )

        eval_row_start = max(
            0,
            int(evaluation_window.row_off)
        )

        eval_col_end = min(
            sar_src.width,
            eval_col_start + int(evaluation_window.width)
        )

        eval_row_end = min(
            sar_src.height,
            eval_row_start + int(evaluation_window.height)
        )

        eval_width = (
            eval_col_end -
            eval_col_start
        )

        eval_height = (
            eval_row_end -
            eval_row_start
        )

        evaluation_window = rasterio.windows.Window(
            eval_col_start,
            eval_row_start,
            eval_width,
            eval_height
        )

        eval_transform = sar_src.window_transform(
            evaluation_window
        )

        print()
        print("Exact evaluation window:")
        print(
            f"  Rows: "
            f"{eval_row_start} -> {eval_row_end}"
        )
        print(
            f"  Cols: "
            f"{eval_col_start} -> {eval_col_end}"
        )
        print(
            f"  Size: "
            f"{eval_width} x {eval_height}"
        )

        # ====================================================
        # EXPANDED PROCESSING WINDOW
        # ====================================================

        margin_x = (
            MARGIN_PIXELS *
            abs(sar_src.transform.a)
        )

        margin_y = (
            MARGIN_PIXELS *
            abs(sar_src.transform.e)
        )

        process_left = (
            ref_bounds.left -
            margin_x
        )

        process_right = (
            ref_bounds.right +
            margin_x
        )

        process_bottom = (
            ref_bounds.bottom -
            margin_y
        )

        process_top = (
            ref_bounds.top +
            margin_y
        )

        processing_window = from_bounds(
            process_left,
            process_bottom,
            process_right,
            process_top,
            transform=sar_src.transform
        )

        processing_window = (
            processing_window
            .round_offsets()
            .round_lengths()
        )

        proc_col_start = max(
            0,
            int(processing_window.col_off)
        )

        proc_row_start = max(
            0,
            int(processing_window.row_off)
        )

        proc_col_end = min(
            sar_src.width,
            proc_col_start +
            int(processing_window.width)
        )

        proc_row_end = min(
            sar_src.height,
            proc_row_start +
            int(processing_window.height)
        )

        proc_width = (
            proc_col_end -
            proc_col_start
        )

        proc_height = (
            proc_row_end -
            proc_row_start
        )

        processing_window = rasterio.windows.Window(
            proc_col_start,
            proc_row_start,
            proc_width,
            proc_height
        )

        print()
        print("Expanded processing window:")
        print(
            f"  Rows: "
            f"{proc_row_start} -> {proc_row_end}"
        )
        print(
            f"  Cols: "
            f"{proc_col_start} -> {proc_col_end}"
        )
        print(
            f"  Size: "
            f"{proc_width} x {proc_height}"
        )

        # ====================================================
        # READ SAR PROCESSING AREA
        # ====================================================

        print()
        print("Reading SAR processing area...")

        change = sar_src.read(
            1,
            window=processing_window
        )

        processing_transform = (
            sar_src.window_transform(
                processing_window
            )
        )

        # ====================================================
        # RAW -2 dB MASK
        # ====================================================

        valid = np.isfinite(change)

        if sar_src.nodata is not None:
            valid &= (
                change != sar_src.nodata
            )

        raw_mask = (
            valid &
            (change <= THRESHOLD_DB)
        )

        raw_pixels = int(
            np.sum(raw_mask)
        )

        print()
        print(
            f"Raw -2 dB pixels: "
            f"{raw_pixels:,}"
        )

        # ====================================================
        # CONNECTED COMPONENT FILTER
        # ====================================================

        print()
        print("Running connected-component analysis...")

        labeled, num_features = ndimage.label(
            raw_mask,
            structure=STRUCTURE
        )

        print(
            f"Connected components: "
            f"{num_features:,}"
        )

        component_sizes = np.bincount(
            labeled.ravel()
        )

        keep = (
            component_sizes >=
            MIN_COMPONENT_PIXELS
        )

        keep[0] = False

        filtered_processing_mask = (
            keep[labeled]
        )

        filtered_pixels = int(
            np.sum(
                filtered_processing_mask
            )
        )

        print(
            f"Filtered processing pixels: "
            f"{filtered_pixels:,}"
        )

        # ====================================================
        # EXTRACT EXACT EVALUATION AREA
        # ====================================================

        row_offset = (
            eval_row_start -
            proc_row_start
        )

        col_offset = (
            eval_col_start -
            proc_col_start
        )

        validated_mask = (
            filtered_processing_mask[
                row_offset:
                row_offset + eval_height,
                col_offset:
                col_offset + eval_width
            ]
        )

        print()
        print("Exact evaluation mask:")
        print(
            f"  Size: "
            f"{validated_mask.shape[1]} x "
            f"{validated_mask.shape[0]}"
        )

        validated_pixels = int(
            np.sum(validated_mask)
        )

        print(
            f"Flood pixels in evaluation area: "
            f"{validated_pixels:,}"
        )

        # ====================================================
        # REPROJECT APSAC REFERENCE
        # ====================================================

        print()
        print(
            "Reprojecting APSAC reference "
            "to exact SAR evaluation grid..."
        )

        reference_on_sar = np.zeros(
            (
                eval_height,
                eval_width
            ),
            dtype=np.uint8
        )

        reproject(
            source=rasterio.band(
                ref_src,
                1
            ),
            destination=reference_on_sar,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            dst_transform=eval_transform,
            dst_crs=sar_src.crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0
        )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    sar_flood = (
        validated_mask == 1
    )

    ref_flood = (
        reference_on_sar == 1
    )

    tp = int(
        np.sum(
            sar_flood &
            ref_flood
        )
    )

    fp = int(
        np.sum(
            sar_flood &
            ~ref_flood
        )
    )

    fn = int(
        np.sum(
            ~sar_flood &
            ref_flood
        )
    )

    tn = int(
        np.sum(
            ~sar_flood &
            ~ref_flood
        )
    )

    precision = (
        tp /
        (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp /
        (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    iou = (
        tp /
        (tp + fp + fn)
        if tp + fp + fn > 0
        else 0.0
    )

    # ========================================================
    # AREA
    # ========================================================

    pixel_area_m2 = (
        abs(eval_transform.a) *
        abs(eval_transform.e)
    )

    flood_area_km2 = (
        validated_pixels *
        pixel_area_m2 /
        1_000_000
    )

    reference_pixels = int(
        np.sum(ref_flood)
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL VALIDATION RESULTS")
    print("=" * 60)

    print()
    print(
        f"Evaluation pixels: "
        f"{eval_width * eval_height:,}"
    )

    print(
        f"APSAC flood pixels: "
        f"{reference_pixels:,}"
    )

    print()
    print(f"TP: {tp:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TN: {tn:,}")

    print()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    print()
    print(
        f"Detected flood area: "
        f"{flood_area_km2:.2f} km²"
    )

    # ========================================================
    # SAVE VALIDATED MASK
    # ========================================================

    print()
    print("Saving validated mask...")

    with rasterio.open(
        CHANGE_MAP
    ) as src:

        profile = src.profile.copy()

    profile.update(
        height=eval_height,
        width=eval_width,
        transform=eval_transform,
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT_MASK,
        "w",
        **profile
    ) as dst:

        dst.write(
            validated_mask.astype(
                np.uint8
            ),
            1
        )

    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    print()
    print("Creating APSAC comparison image...")

    scale = max(
        eval_height / MAX_DISPLAY_SIZE,
        eval_width / MAX_DISPLAY_SIZE
    )

    step = max(
        1,
        int(np.ceil(scale))
    )

    sar_small = sar_flood[
        ::step,
        ::step
    ]

    ref_small = ref_flood[
        ::step,
        ::step
    ]

    rgb = np.ones(
        (
            sar_small.shape[0],
            sar_small.shape[1],
            3
        ),
        dtype=np.float32
    )

    ref_only = (
        ref_small &
        ~sar_small
    )

    sar_only = (
        sar_small &
        ~ref_small
    )

    both = (
        sar_small &
        ref_small
    )

    # APSAC only
    rgb[ref_only] = [
        1.0,
        0.0,
        0.0
    ]

    # SAR only
    rgb[sar_only] = [
        0.0,
        0.3,
        1.0
    ]

    # Both
    rgb[both] = [
        1.0,
        0.0,
        1.0
    ]

    plt.figure(
        figsize=(12, 10)
    )

    plt.imshow(
        rgb,
        interpolation="nearest"
    )

    plt.title(
        "Validated SAR Flood Mask vs APSAC Reference\n"
        "-2 dB + 1000 Pixel Component Filter"
    )

    plt.axis("off")

    legend_items = [
        Patch(
            facecolor="red",
            label="APSAC reference only"
        ),
        Patch(
            facecolor="blue",
            label="SAR detected only"
        ),
        Patch(
            facecolor="magenta",
            label="Both detected"
        ),
        Patch(
            facecolor="white",
            edgecolor="black",
            label="Neither"
        )
    ]

    plt.legend(
        handles=legend_items,
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_IMAGE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print()
    print("Validated mask:")
    print(OUTPUT_MASK)

    print()
    print("Comparison image:")
    print(OUTPUT_IMAGE)


if __name__ == "__main__":
    main()