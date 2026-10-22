import os
import csv
import requests

from PIL import Image
from io import BytesIO

from tile_utils import latlon_to_tile, tile_to_latlon

# Krishna River (Vijayawada)
latitude = 16.5062
longitude = 80.6480
zoom = 15

center_x, center_y = latlon_to_tile(latitude, longitude, zoom)

os.makedirs("../dataset", exist_ok=True)

csv_file = open("../dataset/metadata.csv", "w", newline="")

writer = csv.writer(csv_file)

writer.writerow([
    "filename",
    "tile_x",
    "tile_y",
    "latitude",
    "longitude",
    "zoom"
])

count = 1

for dx in range(-2, 3):
    for dy in range(-2, 3):

        x = center_x + dx
        y = center_y + dy

        url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

        print(f"Downloading Tile {count}")

        response = requests.get(url)

        if response.status_code == 200:

            image = Image.open(BytesIO(response.content))

            filename = f"tile_{count}.jpg"

            image.save(f"../dataset/{filename}")

            tile_lat, tile_lon = tile_to_latlon(x, y, zoom)

            writer.writerow([
                filename,
                x,
                y,
                tile_lat,
                tile_lon,
                zoom
            ])

            print(f"Saved {filename}")

            count += 1

csv_file.close()

print("✅ Dataset + Metadata generated successfully!")