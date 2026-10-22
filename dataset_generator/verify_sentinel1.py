import rasterio
import os

files = [
    "dataset/temporal/krishna_flood_2024/before_20240820/sentinel1_2024-08-20.tif",
    "dataset/temporal/krishna_flood_2024/flood_20240901/sentinel1_2024-09-01.tif"
]

for file_path in files:

    print("\n" + "=" * 60)
    print("Checking:", file_path)
    print("=" * 60)

    if not os.path.exists(file_path):
        print("❌ File not found")
        continue

    size_gb = os.path.getsize(file_path) / (1024 ** 3)
    print(f"File size: {size_gb:.2f} GB")

    try:
        with rasterio.open(file_path) as src:

            print("✅ GeoTIFF opened successfully")
            print("Width:", src.width)
            print("Height:", src.height)
            print("Bands:", src.count)
            print("Data type:", src.dtypes[0])
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)

    except Exception as e:
        print("❌ Could not open file")
        print("Error:", e)

print("\nVerification completed.")