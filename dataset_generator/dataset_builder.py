import os
import sys
import csv
import requests

from io import BytesIO
from PIL import Image

from tile_utils import latlon_to_tile, tile_to_latlon

# =====================================================
# Import locations from documentation folder
# =====================================================

sys.path.append("../documentation")
from locations import LOCATIONS

# =====================================================
# CONFIGURATION
# =====================================================

ZOOM = 18
GRID_SIZE = 3

DATASET_FOLDER = "../dataset/barren_v2"
METADATA_FILE = "../dataset/barren_metadata_v2.csv"

# =====================================================
# Create Metadata File
# =====================================================

os.makedirs(DATASET_FOLDER, exist_ok=True)

file_exists = os.path.exists(METADATA_FILE)

processed_locations = set()

if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            processed_locations.add(
                (row["category"], row["location"])
            )

csv_file = open(METADATA_FILE, "a", newline="")

writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow([
        "filename",
        "category",
        "location",
        "latitude",
        "longitude",
        "tile_x",
        "tile_y",
        "zoom"
    ])

# =====================================================
# Download Images
# =====================================================

half = GRID_SIZE // 2

for category, place_name, latitude, longitude in LOCATIONS:

    if (category, place_name) in processed_locations:
        print(f"Skipping {place_name} (already downloaded)")
        continue

    print("\n========================================")
    print(f"Category : {category}")
    print(f"Location : {place_name}")
    print("========================================")

    print("\n========================================")
    print(f"Category : {category}")
    print(f"Location : {place_name}")
    print("========================================")

    save_folder = os.path.join(DATASET_FOLDER, category)

    os.makedirs(save_folder, exist_ok=True)

    center_x, center_y = latlon_to_tile(latitude, longitude, ZOOM)

    image_number = len(os.listdir(save_folder)) + 1

    for dx in range(-half, half + 1):

        for dy in range(-half, half + 1):

            tile_x = center_x + dx
            tile_y = center_y + dy

            url = (
                f"https://services.arcgisonline.com/"
                f"ArcGIS/rest/services/World_Imagery/"
                f"MapServer/tile/{ZOOM}/{tile_y}/{tile_x}"
            )

            print(f"Downloading Tile ({tile_x}, {tile_y})")

            response = requests.get(url, timeout=30)

            if response.status_code == 200:

                image = Image.open(BytesIO(response.content))

                filename = f"{category.lower()}_{image_number}.jpg"

                image.save(os.path.join(save_folder, filename))

                lat, lon = tile_to_latlon(tile_x, tile_y, ZOOM)

                writer.writerow([
                    filename,
                    category,
                    place_name,
                    lat,
                    lon,
                    tile_x,
                    tile_y,
                    ZOOM
                ])

                print("Saved :", filename)

                image_number += 1

            else:
                print("Failed :", tile_x, tile_y)

csv_file.close()

print("\n========================================")
print("Dataset generation completed successfully!")
print("Metadata updated.")
print("========================================")