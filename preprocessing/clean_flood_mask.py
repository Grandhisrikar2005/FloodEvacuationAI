
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FOLDER = os.path.join(
    PROJECT_ROOT,
    "dataset", "temporal", "krishna_flood_2024",
    "flood_analysis"
)

INPUT_PATH = os.path.join(
    FOLDER, "flood_inundation_mask_v2_2024-09-01.tif"
)

OUTPUT_PATH = os.path.join(
    FOLDER, "cleaned_flood_mask_v2_2024-09-01.tif"
)

MIN_REGION_PIXELS = 100
WINDOW_SIZE = 1024


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"V2 input mask not found: {INPUT_PATH}"
        )

    print("FLOOD MASK CLEANING V2")
    print("Input:", INPUT_PATH)
    print("Minimum region size:", MIN_REGION_PIXELS, "pixels")
    print("Using memory-safe window processing.")

    # Windowed local cleanup avoids loading the huge raster
    # and its connected-component labels into RAM.
    # Each window is processed independently, so regions
    # crossing window boundaries may be split.
    with rasterio.open(INPUT_PATH) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            nodata=255,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        total_before = 0
        total_after = 0
        total_nodata = 0

        with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
            for row in range(0, src.height, WINDOW_SIZE):
                for col in range(0, src.width, WINDOW_SIZE):
                    window = Window(
                        col,
                        row,
                        min(WINDOW_SIZE, src.width - col),
                        min(WINDOW_SIZE, src.height - row)
                    )

                    mask = src.read(1, window=window)
                    valid = mask != 255
                    flood = valid & (mask == 1)

                    total_before += int(flood.sum())
                    total_nodata += int((~valid).sum())

                    # Remove small connected regions inside each window.
                    labels, count = ndimage.label(
                        flood,
                        structure=np.ones((3, 3), dtype=np.uint8)
                    )

                    if count:
                        sizes = np.bincount(labels.ravel())
                        keep = sizes >= MIN_REGION_PIXELS
                        keep[0] = False
                        cleaned = keep[labels] & valid
                    else:
                        cleaned = np.zeros(mask.shape, dtype=bool)

                    output = np.zeros(mask.shape, dtype=np.uint8)
                    output[cleaned] = 1
                    output[~valid] = 255

                    total_after += int(cleaned.sum())
                    dst.write(output, 1, window=window)

                print(
                    f"Processed rows up to "
                    f"{min(row + WINDOW_SIZE, src.height)} "
                    f"of {src.height}"
                )

    print("\nCLEANING COMPLETE")
    print("Flood candidates before:", f"{total_before:,}")
    print("Candidates after:", f"{total_after:,}")
    print("Candidates removed:", f"{total_before - total_after:,}")
    print("NoData pixels:", f"{total_nodata:,}")
    print("Output:", OUTPUT_PATH)

    with rasterio.open(OUTPUT_PATH) as result:
        print("CRS:", result.crs)
        print("NoData value:", result.nodata)
        print("Output encoding: 0=non-candidate, 1=candidate, 255=NoData")


if __name__ == "__main__":
    main()
