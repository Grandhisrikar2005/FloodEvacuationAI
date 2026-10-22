
import os
import numpy as np
import rasterio

MASK_PATH = (
    "dataset/temporal/krishna_flood_2024/"
    "flood_analysis/validated_flood_mask_2024-09-01.tif"
)


def main():
    if not os.path.exists(MASK_PATH):
        raise FileNotFoundError(f"Mask not found: {MASK_PATH}")

    with rasterio.open(MASK_PATH) as src:
        print("\n=== FLOOD MASK INSPECTION ===")
        print("File:", MASK_PATH)
        print("Dimensions:", src.width, "x", src.height)
        print("Bands:", src.count)
        print("Coordinate reference system:", src.crs)
        print("Pixel size:", src.res)
        print("Bounds:", src.bounds)
        print("Data type:", src.dtypes)
        print("NoData value:", src.nodata)

        for band in range(1, src.count + 1):
            data = src.read(band, masked=True)
            valid = data.compressed()

            print(f"\nBand {band}")
            print("Valid pixels:", valid.size)

            if valid.size:
                values, counts = np.unique(valid, return_counts=True)
                print("Unique values:", values[:30])
                print("Value counts:")

                for value, count in zip(values[:30], counts[:30]):
                    print(f"  {value}: {count}")

                if len(values) > 30:
                    print("  ... additional values omitted")

        print("\nInspection complete. No files were modified.")


if __name__ == "__main__":
    main()
