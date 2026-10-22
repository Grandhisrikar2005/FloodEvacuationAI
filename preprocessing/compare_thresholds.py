import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CHANGE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_analysis",
    "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_analysis",
    "threshold_comparison.png"
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    -3.0,
    -5.0,
    -7.0,
    -9.0
]


# ============================================================
# VISUALIZATION SIZE
# ============================================================

MAX_SIZE = 1000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FLOOD THRESHOLD SPATIAL COMPARISON")
    print("=" * 70)

    with rasterio.open(CHANGE_PATH) as src:

        print("\nInput CRS:", src.crs)
        print("Original size:", src.width, "x", src.height)
        print("Resolution:", src.res)

        # ----------------------------------------------------
        # Calculate reduced visualization size
        # ----------------------------------------------------

        scale = min(
            MAX_SIZE / src.width,
            MAX_SIZE / src.height
        )

        out_width = max(
            1,
            int(src.width * scale)
        )

        out_height = max(
            1,
            int(src.height * scale)
        )

        print(
            "Visualization size:",
            out_width,
            "x",
            out_height
        )

        # ----------------------------------------------------
        # Read reduced SAR change image
        # ----------------------------------------------------

        change = src.read(
            1,
            out_shape=(
                out_height,
                out_width
            ),
            resampling=rasterio.enums.Resampling.bilinear
        )

        nodata = src.nodata

    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    axes = axes.ravel()

    for i, threshold in enumerate(THRESHOLDS):

        print(
            f"\nCreating threshold: {threshold} dB"
        )

        # ----------------------------------------------------
        # Create binary flood mask
        # ----------------------------------------------------

        valid = (
            np.isfinite(change)
            &
            (change != nodata)
        )

        flood_mask = (
            valid
            &
            (change <= threshold)
        )

        # ----------------------------------------------------
        # Percentage in visualization
        # ----------------------------------------------------

        valid_count = np.count_nonzero(valid)

        flood_count = np.count_nonzero(
            flood_mask
        )

        if valid_count > 0:
            percentage = (
                flood_count /
                valid_count *
                100
            )
        else:
            percentage = 0

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        axes[i].imshow(
            flood_mask,
            cmap="gray",
            interpolation="nearest"
        )

        axes[i].set_title(
            f"{threshold:.0f} dB threshold\n"
            f"Detected: {percentage:.2f}%"
        )

        axes[i].set_xlabel("Pixel Column")
        axes[i].set_ylabel("Pixel Row")

        print(
            f"Detected in visualization: "
            f"{percentage:.2f}%"
        )

    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        "Sentinel-1 SAR Flood Threshold Comparison\n"
        "20 August 2024 → 1 September 2024",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # ========================================================
    # SAVE
    # ========================================================

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()