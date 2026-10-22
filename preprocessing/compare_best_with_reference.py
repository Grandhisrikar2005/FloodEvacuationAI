
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds, Window


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "dataset" / "temporal" / "krishna_flood_2024"
ANALYSIS = BASE / "flood_analysis"
REFERENCE_DIR = BASE / "reference"

# Evaluate the globally merged V3 candidate mask.
PREDICTION_PATH = (
    ANALYSIS / "best_flood_mask_v3_2024-09-01.tif"
)

REFERENCE_PATH = (
    REFERENCE_DIR / "reference_flood_mask_v3_2024-09-01.tif"
)

OUTPUT_PATH = (
    ANALYSIS / "best_v3_vs_apsac_reference_v3.png"
)

MARGIN_PIXELS = 300


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, reference, valid):
    pred_positive = (prediction == 1) & valid
    ref_positive = (reference == 1) & valid

    tp = int(np.count_nonzero(pred_positive & ref_positive))
    fp = int(np.count_nonzero(pred_positive & ~ref_positive & valid))
    fn = int(np.count_nonzero(~pred_positive & ref_positive & valid))
    tn = int(np.count_nonzero(~pred_positive & ~ref_positive & valid))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SAR V3 VS APSAC REFERENCE V3")
    print("=" * 70)

    for path in (PREDICTION_PATH, REFERENCE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    with rasterio.open(PREDICTION_PATH) as pred_src, \
            rasterio.open(REFERENCE_PATH) as ref_src:

        print("\nPREDICTION")
        print("File:", PREDICTION_PATH)
        print("CRS:", pred_src.crs)
        print("Dimensions:", pred_src.width, "x", pred_src.height)
        print("NoData:", pred_src.nodata)

        print("\nREFERENCE")
        print("File:", REFERENCE_PATH)
        print("CRS:", ref_src.crs)
        print("Dimensions:", ref_src.width, "x", ref_src.height)
        print("NoData:", ref_src.nodata)

        if pred_src.crs is None or ref_src.crs is None:
            raise ValueError("Both rasters must have a valid CRS.")

        ref_bounds = rasterio.warp.transform_bounds(
            ref_src.crs,
            pred_src.crs,
            *ref_src.bounds,
            densify_pts=21,
        )

        pixel_x = abs(pred_src.transform.a)
        pixel_y = abs(pred_src.transform.e)

        left = ref_bounds[0] - MARGIN_PIXELS * pixel_x
        bottom = ref_bounds[1] - MARGIN_PIXELS * pixel_y
        right = ref_bounds[2] + MARGIN_PIXELS * pixel_x
        top = ref_bounds[3] + MARGIN_PIXELS * pixel_y

        raw_window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=pred_src.transform,
        ).round_offsets().round_lengths()

        col_start = max(0, int(raw_window.col_off))
        row_start = max(0, int(raw_window.row_off))

        col_end = min(
            pred_src.width,
            int(raw_window.col_off + raw_window.width),
        )
        row_end = min(
            pred_src.height,
            int(raw_window.row_off + raw_window.height),
        )

        if col_end <= col_start or row_end <= row_start:
            raise ValueError(
                "Prediction and reference do not overlap. "
                "Check their CRS and geographic alignment."
            )

        window = Window(
            col_start,
            row_start,
            col_end - col_start,
            row_end - row_start,
        )

        prediction = pred_src.read(1, window=window)
        prediction_valid = pred_src.read_masks(1, window=window) > 0
        prediction_transform = pred_src.window_transform(window)

        reference = np.zeros(prediction.shape, dtype=np.uint8)
        reference_coverage = np.zeros(
            prediction.shape, dtype=np.uint8
        )

        reproject(
            source=rasterio.band(ref_src, 1),
            destination=reference,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=ref_src.nodata,
            dst_transform=prediction_transform,
            dst_crs=pred_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        # The reference mask has no NoData metadata.
        # Treat its raster footprint as classified.
        source_footprint = np.ones(
            (ref_src.height, ref_src.width),
            dtype=np.uint8,
        )

        reproject(
            source=source_footprint,
            destination=reference_coverage,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=None,
            dst_transform=prediction_transform,
            dst_crs=pred_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    prediction_valid &= np.isin(prediction, [0, 1])

    valid = (
        prediction_valid
        & (reference_coverage == 1)
        & np.isin(reference, [0, 1])
    )

    if not np.any(valid):
        raise ValueError("No valid overlapping pixels were found.")

    metrics = calculate_metrics(prediction, reference, valid)

    print("\n" + "=" * 70)
    print("CONFUSION COUNTS")
    print("=" * 70)
    print("Valid overlapping pixels:", f"{int(valid.sum()):,}")
    print("True positives:", f"{metrics['tp']:,}")
    print("False positives:", f"{metrics['fp']:,}")
    print("False negatives:", f"{metrics['fn']:,}")
    print("True negatives:", f"{metrics['tn']:,}")

    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 score:  {metrics['f1']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")

    # Downsample only for visualization, not metric calculations.
    stride = max(1, int(np.ceil(max(prediction.shape) / 1800)))

    p = prediction[::stride, ::stride]
    r = reference[::stride, ::stride]
    v = valid[::stride, ::stride]

    pred_image = np.full(p.shape, 2, dtype=np.uint8)
    ref_image = np.full(r.shape, 2, dtype=np.uint8)

    pred_image[v] = p[v]
    ref_image[v] = r[v]

    # 0=TN, 1=TP, 2=FP, 3=FN, 4=invalid.
    agreement = np.full(p.shape, 4, dtype=np.uint8)
    agreement[v & (p == 0) & (r == 0)] = 0
    agreement[v & (p == 1) & (r == 1)] = 1
    agreement[v & (p == 1) & (r == 0)] = 2
    agreement[v & (p == 0) & (r == 1)] = 3

    binary_cmap = ListedColormap([
        "#f2f2f2", "#00cfe8", "#808080"
    ])
    agreement_cmap = ListedColormap([
        "#f2f2f2", "#00a650", "#ff3030",
        "#2864dc", "#808080"
    ])

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))

    axes[0].imshow(
        pred_image, cmap=binary_cmap, vmin=0, vmax=2,
        interpolation="nearest",
    )
    axes[0].set_title("SAR candidate mask V3")

    axes[1].imshow(
        ref_image, cmap=binary_cmap, vmin=0, vmax=2,
        interpolation="nearest",
    )
    axes[1].set_title("APSAC cyan-derived reference V3")

    axes[2].imshow(
        agreement, cmap=agreement_cmap, vmin=0, vmax=4,
        interpolation="nearest",
    )
    axes[2].set_title(
        "Agreement / disagreement\n"
        "Green=TP, Red=FP, Blue=FN"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "SAR V3 vs APSAC Reference V3\n"
        f"Precision={metrics['precision']:.3f} | "
        f"Recall={metrics['recall']:.3f} | "
        f"F1={metrics['f1']:.3f} | IoU={metrics['iou']:.3f}"
    )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nComparison image saved:")
    print(OUTPUT_PATH)

    print(
        "\nIMPORTANT: The APSAC reference is extracted from map colors. "
        "These metrics measure agreement with that extracted map, "
        "not independently verified flood-detection accuracy."
    )


if __name__ == "__main__":
    main()
