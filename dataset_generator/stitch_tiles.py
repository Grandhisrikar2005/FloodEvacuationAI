from PIL import Image
import os

tile_size = 256
grid_size = 5

canvas = Image.new("RGB", (tile_size * grid_size, tile_size * grid_size))

count = 1

for row in range(grid_size):
    for col in range(grid_size):

        filename = f"../dataset/tile_{count}.jpg"

        if os.path.exists(filename):

            tile = Image.open(filename)

            canvas.paste(tile, (col * tile_size, row * tile_size))

        count += 1

canvas.save("../outputs/krishna_area.jpg")

canvas.show()

print("✅ Area stitched successfully!")