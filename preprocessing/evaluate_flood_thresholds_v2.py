
from pathlib import Path
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds, Window


# ============================================================
# PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "dataset" / "temporal" / "krishna_flood_2024"
SAR_PATH = (
    BASE / "flood_analysis"
    / "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)
REFERENCE_PATH = (
    BASE / "reference"
    / "reference_flood_mask_v3_2024-09-01.tif"
)

OUTPUT_DIR = BASE / "flood_analysis"
CSV_PATH = OUTPUT_DIR / "threshold_evaluation_v2.csv"
PLOT_PATH = OUTPUT_DIR / "threshold_evaluation_v2.png"

THRESHOLDS_DB = [-1.0, -2.0, -3.0, -4.0, -5.0]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, ref, valid):
    pred_positive = (pred <= 0)  # overwritten below; keeps no assumptions
    pred_positive = pred & valid
    ref_positive = (ref == 1) & valid

    tp = int(np.count_nonzero(pred_positive & ref_positive))
    fp = int(np.count_nonzero(pred_positive & ~ref_positive & valid))
    fn = int(np.count_nonzero(~pred_positive & ref_positive & valid))
    tn = int(np.count_nonzero(~pred_positive & ~ref_positive & valid))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
    )

    return tp, fp, fn, tn, precision, recall, f1, iou, accuracy


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (SAR_PATH, REFERENCE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    print("=" * 70)
    print("SAR THRESHOLD EVALUATION V2")
    print("=" * 70)
    print("SAR:", SAR_PATH)
    print("Reference:", REFERENCE_PATH)
    print("Thresholds:", THRESHOLDS_DB)

    with rasterio.open(SAR_PATH) as sar, \
            rasterio.open(REFERENCE_PATH) as ref_src:

        if sar.crs is None or ref_src.crs is None:
            raise ValueError("Both rasters must have a CRS.")

        # Transform reference bounds into the SAR coordinate system.
        bounds = transform_bounds(
            ref_src.crs,
            sar.crs,
            *ref_src.bounds,
            densify_pts=21
        )

        raw_window = from_bounds(
            *bounds,
            transform=sar.transform
        ).round_offsets().round_lengths()

        col0 = max(0, int(raw_window.col_off))
        row0 = max(0, int(raw_window.row_off))
        col1 = min(
            sar.width,
            int(raw_window.col_off + raw_window.width)
        )
        row1 = min(
            sar.height,
            int(raw_window.row_off + raw_window.height)
        )

        if col1 <= col0 or row1 <= row0:
            raise ValueError("SAR and reference have no overlapping area.")

        window = Window(
            col0, row0, col1 - col0, row1 - row0
        )
        transform = sar.window_transform(window)

        print("\nReading SAR overlap only...")
        change = sar.read(1, window=window)
        sar_valid = sar.read_masks(1, window=window) > 0
        sar_valid &= np.isfinite(change)

        if sar.nodata is not None:
            sar_valid &= change != sar.nodata

        reference = np.zeros(change.shape, dtype=np.uint8)
        coverage = np.zeros(change.shape, dtype=np.uint8)

        # Reproject the reference labels onto the SAR grid.
        reproject(
            source=rasterio.band(ref_src, 1),
            destination=reference,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=ref_src.nodata,
            dst_transform=transform,
            dst_crs=sar.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        # Explicitly track the reference footprint separately.
        footprint = np.ones(
            (ref_src.height, ref_src.width), dtype=np.uint8
        )

        reproject(
            source=footprint,
            destination=coverage,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=None,
            dst_transform=transform,
            dst_crs=sar.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        valid = (
            sar_valid
            & (coverage == 1)
            & np.isin(reference, [0, 1])
        )

        pixel_area_m2 = abs(
            sar.transform.a * sar.transform.e
            - sar.transform.b * sar.transform.d
        )

    if not valid.any():
        raise ValueError("No valid overlapping pixels found.")

    reference_positive = int(np.count_nonzero((reference == 1) & valid))
    reference_negative = int(np.count_nonzero((reference == 0) & valid))

    print("\nVALIDATION AREA")
    print("-" * 45)
    print("Valid pixels:", f"{int(valid.sum()):,}")
    print("Reference positive pixels:", f"{reference_positive:,}")
    print("Reference negative pixels:", f"{reference_negative:,}")
    print("Pixel area:", f"{pixel_area_m2:.2f} m²")

    rows = []

    for threshold in THRESHOLDS_DB:
        # Evaluate the raw SAR threshold only.
        # No connected-component filtering is applied here.
        prediction = change <= threshold

        metrics = calculate_metrics(
            prediction,
            reference,
            valid
        )

        tp, fp, fn, tn, precision, recall, f1, iou, accuracy = metrics
        candidate_pixels = int(np.count_nonzero(prediction & valid))
        candidate_area_km2 = (
            candidate_pixels * pixel_area_m2 / 1_000_000
        )

        row = {
            "threshold_db": threshold,
            "valid_pixels": int(valid.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "candidate_pixels": candidate_pixels,
            "candidate_area_km2": candidate_area_km2,
        }
        rows.append(row)

        print(f"\nThreshold: {threshold:.1f} dB")
        print(
            f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}"
        )
        print(
            f"  Precision={precision:.4f}  Recall={recall:.4f}"
        )
        print(
            f"  F1={f1:.4f}  IoU={iou:.4f}  Accuracy={accuracy:.4f}"
        )
        print(
            f"  Candidates={candidate_pixels:,}  "
            f"Area={candidate_area_km2:.2f} km²"
        )

    # Save detailed results.
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Plot metrics against threshold.
    thresholds = [r["threshold_db"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        thresholds, [r["precision"] for r in rows],
        marker="o", label="Precision"
    )
    axes[0].plot(
        thresholds, [r["recall"] for r in rows],
        marker="o", label="Recall"
    )
    axes[0].plot(
        thresholds, [r["f1"] for r in rows],
        marker="o", label="F1"
    )
    axes[0].plot(
        thresholds, [r["iou"] for r in rows],
        marker="o", label="IoU"
    )
    axes[0].set_xlabel("SAR change threshold (dB)")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Threshold vs. validation metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        thresholds,
        [r["candidate_area_km2"] for r in rows],
        marker="o"
    )
    axes[1].set_xlabel("SAR change threshold (dB)")
    axes[1].set_ylabel("Candidate area (km²)")
    axes[1].set_title("Threshold vs. candidate area")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    best = max(rows, key=lambda item: item["f1"])

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("Highest observed F1 threshold:", best["threshold_db"], "dB")
    print(f"F1: {best['f1']:.4f}")
    print("CSV:", CSV_PATH)
    print("Plot:", PLOT_PATH)
    print(
        "\nCAUTION: The highest F1 is only the best match to the "
        "current cyan-derived reference among the tested thresholds. "
        "It is not independent proof of flood-detection accuracy."
    )


if __name__ == "__main__":
    main()
