
import os
import numpy as np
import rasterio

PATH = (
    "dataset/temporal/krishna_flood_2024/flood_analysis/"
    "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)


def main():
    if not os.path.exists(PATH):
        raise FileNotFoundError(PATH)

    with rasterio.open(PATH) as src:
        print("=== SAR CHANGE RASTER DIAGNOSTICS ===")
        print("Dimensions:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Resolution:", src.res)
        print("Data type:", src.dtypes[0])
        print("NoData:", src.nodata)
        print("Scale:", src.scales)
        print("Offset:", src.offsets)
        print("Units:", src.units)
        print("Bounds:", src.bounds)

        # Sample up to 8 blocks across the raster.
        # This is diagnostic sampling, not full-raster statistics.
        samples = []

        for index, (_, window) in enumerate(src.block_windows(1)):
            if index >= 8:
                break

            data = src.read(1, window=window, masked=True)
            values = data.compressed()
            if values.size:
                samples.append(values)

        if not samples:
            print("No valid values found in sampled blocks.")
            return

        values = np.concatenate(samples)

        print("\nSampled valid pixel count:", values.size)
        print("Sample minimum:", float(values.min()))
        print("Sample maximum:", float(values.max()))
        print(
            "Sample quantiles:",
            np.quantile(values, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]).tolist()
        )

        print("\nSample threshold counts:")
        for threshold in [-3, -2, -1, 0, 1, 2, 3]:
            count = int(np.count_nonzero(values <= threshold))
            print(
                f"Values <= {threshold}: "
                f"{count:,} ({count / values.size * 100:.2f}%)"
            )

        print(
            "\nNOTE: Statistics above cover only sampled blocks, "
            "not the entire raster."
        )


if __name__ == "__main__":
    main()
