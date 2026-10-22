
import os
import numpy as np
import rasterio
from rasterio.windows import Window

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset", "temporal", "krishna_flood_2024",
    "flood_analysis"
)

CHANGE_PATH = os.path.join(
    BASE_DIR,
    "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "flood_inundation_mask_v2_2024-09-01.tif"
)

THRESHOLD_DB = -3.0
WINDOW_SIZE = 1024
NODATA_VALUE = 255


def main():
    print("=" * 60)
    print("FLOOD-CHANGE CANDIDATE MASK V2")
    print("=" * 60)
    print("Threshold:", THRESHOLD_DB, "dB")

    valid_pixels = 0
    candidate_pixels = 0
    invalid_pixels = 0

    with rasterio.open(CHANGE_PATH) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=NODATA_VALUE,
            compress="deflate",
            predictor=2,
            tiled=True,
            BIGTIFF="YES",
        )

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            total_blocks = (
                (src.height + WINDOW_SIZE - 1) // WINDOW_SIZE
                * (src.width + WINDOW_SIZE - 1) // WINDOW_SIZE
            )
            completed = 0

            for row in range(0, src.height, WINDOW_SIZE):
                for col in range(0, src.width, WINDOW_SIZE):
                    height = min(WINDOW_SIZE, src.height - row)
                    width = min(WINDOW_SIZE, src.width - col)
                    window = Window(col, row, width, height)

                    # Masked read respects the source NoData metadata.
                    source = src.read(1, window=window, masked=True)
                    change = source.data
                    valid = (
                        ~np.ma.getmaskarray(source)
                        & np.isfinite(change)
                    )

                    result = np.full(
                        change.shape, NODATA_VALUE, dtype=np.uint8
                    )

                    result[valid] = 0
                    candidates = valid & (change <= THRESHOLD_DB)
                    result[candidates] = 1

                    valid_pixels += int(np.count_nonzero(valid))
                    candidate_pixels += int(np.count_nonzero(candidates))
                    invalid_pixels += int(valid.size - np.count_nonzero(valid))

                    dst.write(result, 1, window=window)

                    completed += 1
                    print(
                        f"\rProcessed {completed}/{total_blocks} blocks",
                        end="",
                    )

    print("\n\nMask saved:", OUTPUT_PATH)
    print("Valid pixels:", f"{valid_pixels:,}")
    print("Candidate pixels:", f"{candidate_pixels:,}")
    print("Invalid pixels:", f"{invalid_pixels:,}")

    if valid_pixels:
        print(
            "Candidate percentage of valid pixels:",
            f"{candidate_pixels / valid_pixels * 100:.2f}%",
        )

    print("\nVerifying output values...")
    with rasterio.open(OUTPUT_PATH) as result:
        counts = {0: 0, 1: 0, NODATA_VALUE: 0}

        for _, window in result.block_windows(1):
            block = result.read(1, window=window)
            values, frequencies = np.unique(block, return_counts=True)

            for value, frequency in zip(values, frequencies):
                if int(value) in counts:
                    counts[int(value)] += int(frequency)

        print("Non-candidate pixels (0):", f"{counts[0]:,}")
        print("Flood-change candidates (1):", f"{counts[1]:,}")
        print("NoData pixels (255):", f"{counts[255]:,}")
        print("CRS:", result.crs)
        print("Bounds:", result.bounds)
        print("NoData metadata:", result.nodata)

    print("\nNOTE: These are SAR-change candidates, not confirmed flood pixels.")
    print("Compare the results with your reference mask before routing.")


if __name__ == "__main__":
    main()
