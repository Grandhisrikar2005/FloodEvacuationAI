
from pathlib import Path
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

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

CSV_PATH = OUTPUT_DIR / "filtered_vs_raw_evaluation.csv"
PLOT_PATH = OUTPUT_DIR / "filtered_vs_raw_evaluation.png"

THRESHOLDS_DB = [-1.0, -2.0, -3.0, -4.0, -5.0]
MIN_COMPONENT_PIXELS = 1000

# 8-connected neighbourhood, consistent with the existing mask script.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(prediction, reference, valid):
    pred_positive = prediction & valid
    ref_positive = (reference == 1) & valid

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

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn else 0.0
    )

    accuracy = (
        (tp + tn) / (tp + tn + fp + fn)
        if tp + tn + fp + fn else 0.0
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
# COMPONENT FILTER
# ============================================================

def remove_small_components(candidate_mask, min_pixels):
    """
    Keep connected candidate regions with at least min_pixels.

    This implementation operates on the reference-overlap crop.
    Regions touching the crop boundary may be truncated, so results
    are exploratory rather than a final full-scene mask.
    """
    labels, number_of_components = ndimage.label(
        candidate_mask,
        structure=STRUCTURE
    )

    if number_of_components == 0:
        return np.zeros(candidate_mask.shape, dtype=bool), 0

    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_pixels
    keep[0] = False

    filtered = keep[labels]

    return filtered, number_of_components


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (SAR_PATH, REFERENCE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    print("=" * 72)
    print("FILTERED VS RAW SAR THRESHOLD EVALUATION")
    print("=" * 72)
    print("SAR:", SAR_PATH)
    print("Reference:", REFERENCE_PATH)
    print("Thresholds:", THRESHOLDS_DB)
    print("Minimum component size:", MIN_COMPONENT_PIXELS)

    # --------------------------------------------------------
    # Read only the SAR region overlapping the reference.
    # --------------------------------------------------------

    with rasterio.open(SAR_PATH) as sar, \
            rasterio.open(REFERENCE_PATH) as ref_src:

        if sar.crs is None or ref_src.crs is None:
            raise ValueError("Both rasters must have a CRS.")

        ref_bounds = transform_bounds(
            ref_src.crs,
            sar.crs,
            *ref_src.bounds,
            densify_pts=21
        )

        raw_window = from_bounds(
            *ref_bounds,
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
            raise ValueError("No overlap between SAR and reference.")

        window = Window(
            col0,
            row0,
            col1 - col0,
            row1 - row0
        )

        sar_transform = sar.window_transform(window)

        print("\nReading SAR overlap...")
        change = sar.read(1, window=window)
        sar_valid = sar.read_masks(1, window=window) > 0
        sar_valid &= np.isfinite(change)

        if sar.nodata is not None:
            sar_valid &= change != sar.nodata

        reference = np.zeros(change.shape, dtype=np.uint8)
        coverage = np.zeros(change.shape, dtype=np.uint8)

        reproject(
            source=rasterio.band(ref_src, 1),
            destination=reference,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=ref_src.nodata,
            dst_transform=sar_transform,
            dst_crs=sar.crs,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

        # Track the reference footprint independently of its labels.
        footprint = np.ones(
            (ref_src.height, ref_src.width),
            dtype=np.uint8
        )

        reproject(
            source=footprint,
            destination=coverage,
            src_transform=ref_src.transform,
            src_crs=ref_src.crs,
            src_nodata=None,
            dst_transform=sar_transform,
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
        raise ValueError("No valid overlapping pixels.")

    print("Valid overlapping pixels:", f"{int(valid.sum()):,}")
    print("Reference positive pixels:", f"{int(np.count_nonzero((reference == 1) & valid)):,}")

    # --------------------------------------------------------
    # Evaluate raw and filtered masks using identical valid pixels.
    # --------------------------------------------------------

    rows = []

    for threshold in THRESHOLDS_DB:
        print("\n" + "-" * 72)
        print(f"Threshold: {threshold:.1f} dB")

        raw_mask = change <= threshold
        raw_mask &= sar_valid

        filtered_mask, component_count = remove_small_components(
            raw_mask,
            MIN_COMPONENT_PIXELS
        )

        raw_metrics = calculate_metrics(
            raw_mask,
            reference,
            valid
        )

        filtered_metrics = calculate_metrics(
            filtered_mask,
            reference,
            valid
        )

        for method, prediction, metrics in (
            ("raw", raw_mask, raw_metrics),
            ("filtered", filtered_mask, filtered_metrics),
        ):
            candidate_pixels = int(
                np.count_nonzero(prediction & valid)
            )

            row = {
                "threshold_db": threshold,
                "method": method,
                "min_component_pixels": (
                    1 if method == "raw" else MIN_COMPONENT_PIXELS
                ),
                "components_before_filter": component_count,
                "valid_pixels": int(valid.sum()),
                **metrics,
                "candidate_pixels": candidate_pixels,
                "candidate_area_km2": (
                    candidate_pixels * pixel_area_m2 / 1_000_000
                ),
            }

            rows.append(row)

            print(
                f"{method.upper():8s} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"IoU={metrics['iou']:.4f} | "
                f"Candidates={candidate_pixels:,} | "
                f"Area={row['candidate_area_km2']:.2f} km²"
            )

    # --------------------------------------------------------
    # Save CSV results.
    # --------------------------------------------------------

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Plot raw versus filtered metrics.
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method in ("raw", "filtered"):
        method_rows = [
            row for row in rows if row["method"] == method
        ]

        thresholds = [
            row["threshold_db"] for row in method_rows
        ]

        axes[0].plot(
            thresholds,
            [row["f1"] for row in method_rows],
            marker="o",
            label=f"{method.title()} F1"
        )

        axes[0].plot(
            thresholds,
            [row["iou"] for row in method_rows],
            marker="x",
            linestyle="--",
            label=f"{method.title()} IoU"
        )

        axes[1].plot(
            thresholds,
            [row["candidate_area_km2"] for row in method_rows],
            marker="o",
            label=method.title()
        )

    axes[0].set_title("F1 and IoU: raw vs filtered")
    axes[0].set_xlabel("SAR threshold (dB)")
    axes[0].set_ylabel("Score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Candidate area: raw vs filtered")
    axes[1].set_xlabel("SAR threshold (dB)")
    axes[1].set_ylabel("Area (km²)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------
    # Report best observed F1 for each method.
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for method in ("raw", "filtered"):
        subset = [row for row in rows if row["method"] == method]
        best = max(subset, key=lambda row: row["f1"])

        print(
            f"{method.title()} best F1: {best['f1']:.4f} "
            f"at {best['threshold_db']:.1f} dB "
            f"(IoU={best['iou']:.4f}, "
            f"precision={best['precision']:.4f}, "
            f"recall={best['recall']:.4f})"
        )

    print("\nCSV:", CSV_PATH)
    print("Plot:", PLOT_PATH)

    print(
        "\nCAUTION: Filtering is evaluated only within the reference crop. "
        "Components crossing the crop boundary may be truncated. "
        "These metrics measure agreement with the cyan-derived map, "
        "not independently verified flood-detection accuracy."
    )


if __name__ == "__main__":
    main()
