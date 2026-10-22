import os
from PIL import Image

INPUT_FOLDER = "../dataset"
OUTPUT_FOLDER = "../dataset/processed"

IMAGE_SIZE = (256, 256)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processed = 0
skipped = 0

for file in os.listdir(INPUT_FOLDER):

    if not file.lower().endswith(".jpg"):
        continue

    if file.startswith("sample"):
        continue

    if file.startswith("krishna"):
        continue

    image_path = os.path.join(INPUT_FOLDER, file)

    try:

        image = Image.open(image_path)

        image = image.convert("RGB")

        image = image.resize(IMAGE_SIZE)

        save_path = os.path.join(OUTPUT_FOLDER, file)

        image.save(save_path)

        processed += 1

        print(f"Processed: {file}")

    except Exception as e:

        skipped += 1

        print(f"Skipped {file}: {e}")

print()
print("===================================")
print("Preprocessing Completed")
print("Processed :", processed)
print("Skipped   :", skipped)
print("===================================")