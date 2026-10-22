import os
import csv
import numpy as np
import rasterio


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
    "threshold_analysis.csv"
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS_DB = [
    -3.0,
    -5.0,
    -7.0,
    -9.0
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SENTINEL-1 FLOOD THRESHOLD ANALYSIS")
    print("=" * 70)

    print("\nInput:")
    print(CHANGE_PATH)

    with rasterio.open(CHANGE_PATH) as src:

        print("\nCRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("NoData:", src.nodata)

        # ----------------------------------------------------
        # Pixel area
        # ----------------------------------------------------

        pixel_area_m2 = (
            abs(src.transform.a) *
            abs(src.transform.e)
        )

        total_pixels = 0
        valid_pixels = 0

        threshold_counts = {
            threshold: 0
            for threshold in THRESHOLDS_DB
        }

        # ----------------------------------------------------
        # Process blocks
        # ----------------------------------------------------

        print("\nAnalyzing change values...")

        total_blocks = 0

        for _, window in src.block_windows(1):
            total_blocks += 1

        processed_blocks = 0

        for _, window in src.block_windows(1):

            data = src.read(
                1,
                window=window
            )

            total_pixels += data.size

            valid = (
                np.isfinite(data)
                &
                (data != src.nodata)
            )

            valid_data = data[valid]

            valid_pixels += valid_data.size

            # ------------------------------------------------
            # Count each threshold
            # ------------------------------------------------

            for threshold in THRESHOLDS_DB:

                count = np.count_nonzero(
                    valid_data <= threshold
                )

                threshold_counts[threshold] += count

            processed_blocks += 1

            if (
                processed_blocks % 50 == 0
                or processed_blocks == total_blocks
            ):

                print(
                    f"\rProcessed: "
                    f"{processed_blocks}/{total_blocks} "
                    f"blocks "
                    f"({processed_blocks / total_blocks * 100:.1f}%)",
                    end=""
                )

    print("\n")

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    print(
        f"\n{'Threshold':<15}"
        f"{'Pixels':<18}"
        f"{'Percentage':<15}"
        f"{'Area (km²)':<15}"
    )

    print("-" * 63)

    results = []

    for threshold in THRESHOLDS_DB:

        count = threshold_counts[threshold]

        percentage = (
            count /
            valid_pixels *
            100
        )

        area_km2 = (
            count *
            pixel_area_m2 /
            1_000_000
        )

        print(
            f"{threshold:<15.1f}"
            f"{count:<18,}"
            f"{percentage:<15.2f}"
            f"{area_km2:<15.2f}"
        )

        results.append({
            "threshold_db": threshold,
            "flood_pixels": count,
            "percentage_valid_pixels": percentage,
            "area_km2": area_km2
        })

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        OUTPUT_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold_db",
                "flood_pixels",
                "percentage_valid_pixels",
                "area_km2"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nValid pixels analyzed: "
        f"{valid_pixels:,}"
    )

    print(
        f"Pixel area: "
        f"{pixel_area_m2:.0f} m²"
    )

    print(
        "\nThe next step will be to compare these "
        "thresholds spatially rather than selecting "
        "one purely from the area numbers."
    )


if __name__ == "__main__":
    main()