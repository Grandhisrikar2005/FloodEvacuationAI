import os
import rasterio

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FILES = [
    os.path.join(
        PROJECT_ROOT,
        "dataset",
        "temporal",
        "krishna_flood_2024",
        "before_20240820",
        "sentinel1_rtc_vv_2024-08-20.tif"
    ),
    os.path.join(
        PROJECT_ROOT,
        "dataset",
        "temporal",
        "krishna_flood_2024",
        "flood_20240901",
        "sentinel1_rtc_vv_2024-09-01.tif"
    )
]

for file_path in FILES:

    print("\n" + "=" * 70)
    print("FILE:")
    print(file_path)
    print("=" * 70)

    if not os.path.exists(file_path):
        print("ERROR: File not found!")
        continue

    file_size = os.path.getsize(file_path)

    print(f"File size: {file_size / (1024 ** 3):.2f} GB")

    try:
        with rasterio.open(file_path) as src:

            print("GeoTIFF: OK")
            print("Width:", src.width)
            print("Height:", src.height)
            print("Bands:", src.count)
            print("Data type:", src.dtypes[0])
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("Resolution:", src.res)
            print("NoData:", src.nodata)
            print("Bounds:", src.bounds)

            # Read a small sample to confirm raster data is readable
            sample = src.read(
                1,
                out_shape=(100, 100)
            )

            print("Sample read: OK")
            print("Sample minimum:", sample.min())
            print("Sample maximum:", sample.max())

    except Exception as e:
        print("ERROR opening GeoTIFF:")
        print(e)

print("\n" + "=" * 70)
print("RTC VERIFICATION COMPLETE")
print("=" * 70)