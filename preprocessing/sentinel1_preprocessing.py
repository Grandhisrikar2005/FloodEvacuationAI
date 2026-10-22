import os
import rasterio


# ==========================================
# INPUT FILES
# ==========================================

BASE_DIR = "dataset/temporal/krishna_flood_2024"

BEFORE_FILE = os.path.join(
    BASE_DIR,
    "before_20240820",
    "sentinel1_2024-08-20.tif"
)

FLOOD_FILE = os.path.join(
    BASE_DIR,
    "flood_20240901",
    "sentinel1_2024-09-01.tif"
)


# ==========================================
# CHECK FILE
# ==========================================

def inspect_file(file_path, label):

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    if not os.path.exists(file_path):
        print("❌ File not found:")
        print(file_path)
        return

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
            print("Transform:", src.transform)

            print("Bounds:", src.bounds)

            print(
                "Resolution:",
                src.res
            )

            print(
                "NoData value:",
                src.nodata
            )

    except Exception as e:

        print("❌ Error reading GeoTIFF")
        print("Error:", e)


# ==========================================
# INSPECT BOTH SCENES
# ==========================================

inspect_file(
    BEFORE_FILE,
    "BEFORE FLOOD — 20 AUGUST 2024"
)

inspect_file(
    FLOOD_FILE,
    "FLOOD PERIOD — 1 SEPTEMBER 2024"
)


print("\n" + "=" * 60)
print("LOCAL FILE INSPECTION COMPLETED")
print("=" * 60)