import requests
from PIL import Image
from io import BytesIO
import os

from tile_utils import latlon_to_tile

# Krishna River near Vijayawada
latitude = 16.5062
longitude = 80.6480
zoom = 15

x, y = latlon_to_tile(latitude, longitude, zoom)

url = f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

print(url)

response = requests.get(url)

if response.status_code == 200:

    image = Image.open(BytesIO(response.content))

    os.makedirs("../dataset", exist_ok=True)

    image.save("../dataset/krishna_tile.jpg")

    image.show()

    print("Downloaded Krishna River tile!")

else:
    print("Failed:", response.status_code)