import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.transform import from_origin


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BEFORE_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "before_20240820",
    "sentinel1_rtc_vv_2024-08-20.tif"
)

FLOOD_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_20240901",
    "sentinel1_rtc_vv_2024-09-01.tif"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "temporal",
    "krishna_flood_2024",
    "flood_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANGE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 1024

# Sentinel-1 RTC values are linear backscatter.
# We calculate:
#
#     change_db = 10 * log10(flood / before)
#
# Negative values mean the radar backscatter decreased.
#
# Water commonly produces a strong decrease in VV
# backscatter compared with land.
#

FLOOD_CHANGE_THRESHOLD_DB = -3.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SENTINEL-1 FLOOD CHANGE DETECTION")
    print("=" * 70)

    print("\nOpening BEFORE scene:")
    print(BEFORE_PATH)

    print("\nOpening FLOOD scene:")
    print(FLOOD_PATH)

    with rasterio.open(BEFORE_PATH) as before, \
         rasterio.open(FLOOD_PATH) as flood:

        # ----------------------------------------------------
        # Basic information
        # ----------------------------------------------------

        print("\n" + "-" * 70)
        print("BEFORE SCENE")
        print("-" * 70)

        print("CRS:", before.crs)
        print("Resolution:", before.res)
        print("Bounds:", before.bounds)
        print("Width:", before.width)
        print("Height:", before.height)

        print("\n" + "-" * 70)
        print("FLOOD SCENE")
        print("-" * 70)

        print("CRS:", flood.crs)
        print("Resolution:", flood.res)
        print("Bounds:", flood.bounds)
        print("Width:", flood.width)
        print("Height:", flood.height)

        # ----------------------------------------------------
        # CRS check
        # ----------------------------------------------------

        if before.crs != flood.crs:
            raise RuntimeError(
                "ERROR: CRS mismatch between the two scenes."
            )

        print("\nCRS check: OK")

        # ----------------------------------------------------
        # Calculate overlapping bounds
        # ----------------------------------------------------

        left = max(before.bounds.left, flood.bounds.left)
        bottom = max(before.bounds.bottom, flood.bounds.bottom)
        right = min(before.bounds.right, flood.bounds.right)
        top = min(before.bounds.top, flood.bounds.top)

        if left >= right or bottom >= top:
            raise RuntimeError(
                "ERROR: The two scenes do not overlap."
            )

        print("\n" + "-" * 70)
        print("COMMON OVERLAPPING AREA")
        print("-" * 70)

        print("Left:", left)
        print("Bottom:", bottom)
        print("Right:", right)
        print("Top:", top)

        # ----------------------------------------------------
        # Convert geographic overlap to pixel coordinates
        # on BEFORE image
        # ----------------------------------------------------

        row_start, col_start = before.index(left, top)
        row_end, col_end = before.index(right, bottom)

        row_start = max(0, row_start)
        col_start = max(0, col_start)

        row_end = min(before.height, row_end)
        col_end = min(before.width, col_end)

        output_height = row_end - row_start
        output_width = col_end - col_start

        if output_width <= 0 or output_height <= 0:
            raise RuntimeError(
                "ERROR: Invalid overlap dimensions."
            )

        # ----------------------------------------------------
        # Output transform
        # ----------------------------------------------------

        output_transform = before.window_transform(
            Window(
                col_start,
                row_start,
                output_width,
                output_height
            )
        )

        print("\n" + "-" * 70)
        print("OUTPUT GRID")
        print("-" * 70)

        print("Width:", output_width)
        print("Height:", output_height)
        print("Resolution:", before.res)
        print("Transform:", output_transform)

        # ----------------------------------------------------
        # Output profile
        # ----------------------------------------------------

        profile = before.profile.copy()

        profile.update(
            driver="GTiff",
            height=output_height,
            width=output_width,
            count=1,
            dtype="float32",
            crs=before.crs,
            transform=output_transform,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="YES"
        )

        # ----------------------------------------------------
        # Create output file
        # ----------------------------------------------------

        print("\nCreating output:")
        print(CHANGE_OUTPUT)

        with rasterio.open(
            CHANGE_OUTPUT,
            "w",
            **profile
        ) as dst:

            total_blocks = (
                ((output_height + WINDOW_SIZE - 1) // WINDOW_SIZE)
                *
                ((output_width + WINDOW_SIZE - 1) // WINDOW_SIZE)
            )

            processed_blocks = 0

            # ------------------------------------------------
            # Process image in windows
            # ------------------------------------------------

            for row in range(
                0,
                output_height,
                WINDOW_SIZE
            ):

                for col in range(
                    0,
                    output_width,
                    WINDOW_SIZE
                ):

                    height = min(
                        WINDOW_SIZE,
                        output_height - row
                    )

                    width = min(
                        WINDOW_SIZE,
                        output_width - col
                    )

                    output_window = Window(
                        col,
                        row,
                        width,
                        height
                    )

                    # ----------------------------------------
                    # Corresponding window in BEFORE image
                    # ----------------------------------------

                    before_window = Window(
                        col_start + col,
                        row_start + row,
                        width,
                        height
                    )

                    before_data = before.read(
                        1,
                        window=before_window
                    ).astype(np.float32)

                    # ----------------------------------------
                    # Destination array for FLOOD scene
                    # ----------------------------------------

                    flood_data = np.full(
                        (height, width),
                        np.nan,
                        dtype=np.float32
                    )

                    # ----------------------------------------
                    # Reproject/resample flood image
                    # onto BEFORE grid
                    # ----------------------------------------

                    destination_transform = before.window_transform(
                        before_window
                    )

                    reproject(
                        source=rasterio.band(flood, 1),
                        destination=flood_data,
                        src_transform=flood.transform,
                        src_crs=flood.crs,
                        dst_transform=destination_transform,
                        dst_crs=before.crs,
                        resampling=Resampling.bilinear,
                        src_nodata=flood.nodata,
                        dst_nodata=np.nan
                    )

                    # ----------------------------------------
                    # Handle NoData
                    # ----------------------------------------

                    before_valid = (
                        before_data != before.nodata
                    )

                    flood_valid = (
                        np.isfinite(flood_data)
                        &
                        (flood_data != flood.nodata)
                    )

                    valid = (
                        before_valid
                        &
                        flood_valid
                        &
                        (before_data > 0)
                        &
                        (flood_data > 0)
                    )

                    # ----------------------------------------
                    # SAR change in dB
                    # ----------------------------------------

                    change_db = np.full(
                        (height, width),
                        -9999.0,
                        dtype=np.float32
                    )

                    change_db[valid] = (
                        10.0
                        *
                        np.log10(
                            flood_data[valid]
                            /
                            before_data[valid]
                        )
                    )

                    # ----------------------------------------
                    # Write block
                    # ----------------------------------------

                    dst.write(
                        change_db,
                        1,
                        window=output_window
                    )

                    processed_blocks += 1

                    print(
                        f"\rProcessing: "
                        f"{processed_blocks}/{total_blocks} "
                        f"blocks "
                        f"({processed_blocks / total_blocks * 100:.1f}%)",
                        end=""
                    )

        print()

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("FLOOD CHANGE OUTPUT CREATED")
    print("=" * 70)

    print("Output:")
    print(CHANGE_OUTPUT)

    with rasterio.open(CHANGE_OUTPUT) as src:

        print("\nOutput CRS:", src.crs)
        print("Output size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes[0])
        print("NoData:", src.nodata)
        print("Bounds:", src.bounds)

        # Read a small sample only
        sample = src.read(
            1,
            out_shape=(100, 100)
        )

        valid_sample = sample[
            sample != src.nodata
        ]

        if len(valid_sample) > 0:

            print(
                "Sample minimum:",
                float(valid_sample.min())
            )

            print(
                "Sample maximum:",
                float(valid_sample.max())
            )

            print(
                "Sample mean:",
                float(valid_sample.mean())
            )

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()