
import os
import numpy as np
import rasterio

BASE_DIR = "dataset/temporal/krishna_flood_2024/flood_analysis"

MASKS = [
    "best_flood_mask_2024-09-01.tif",
    "cleaned_flood_mask_2024-09-01.tif",
    "flood_inundation_mask_2024-09-01.tif",
    "validated_flood_mask_2024-09-01.tif",
]


def inspect_mask(filename):
    path = os.path.join(BASE_DIR, filename)

    print(f"\n{'=' * 60}")
    print(filename)

    with rasterio.open(path) as src:
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodata)
        print("Data type:", src.dtypes[0])

        band = src.read(1, masked=True)
        values = band.compressed()

        if values.size == 0:
            print("No valid pixels found.")
            return

        unique, counts = np.unique(values, return_counts=True)
        print("Unique values:", unique[:20])

        for value, count in zip(unique[:20], counts[:20]):
            percentage = count / values.size * 100
            print(
                f"Value {value}: {count:,} pixels "
                f"({percentage:.2f}%)"
            )


if __name__ == "__main__":
    for filename in MASKS:
        try:
            inspect_mask(filename)
        except Exception as error:
            print(f"Could not inspect {filename}: {error}")
