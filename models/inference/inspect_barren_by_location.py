
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "dataset" / "barren_v2" / "Barren"
CSV_PATH = ROOT / "dataset" / "barren_metadata_v2.csv"
OUTPUT_PATH = ROOT / "dataset" / "barren_by_location_audit.jpg"

metadata = pd.read_csv(CSV_PATH)

columns = 3
cell_width = 360
image_height = 220
label_height = 65
group_gap = 35

font = ImageFont.load_default()

groups = list(metadata.groupby("location", sort=True))
rows_per_group = (9 + columns - 1) // columns

sheet_height = len(groups) * (
    rows_per_group * (image_height + label_height) + group_gap
)

sheet = Image.new(
    "RGB",
    (columns * cell_width, sheet_height),
    "white",
)
draw = ImageDraw.Draw(sheet)

y = 0

for location, group in groups:
    draw.text((10, y + 5), f"{location} ({len(group)} images)",
              fill="black", font=font)
    y += 25

    for index, (_, row) in enumerate(group.iterrows()):
        filename = row["filename"]
        path = IMAGE_DIR / filename

        x = (index % columns) * cell_width
        row_y = y + (index // columns) * (image_height + label_height)

        if not path.exists():
            draw.text((x + 8, row_y + 8),
                      f"MISSING: {filename}", fill="red", font=font)
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((cell_width - 12, image_height - 12))

            paste_x = x + (cell_width - image.width) // 2
            paste_y = row_y + (image_height - image.height) // 2

            sheet.paste(image, (paste_x, paste_y))

        draw.text(
            (x + 8, row_y + image_height + 3),
            f"{filename}\n{location}",
            fill="black",
            font=font,
        )

    y += rows_per_group * (image_height + label_height) + group_gap

sheet.save(OUTPUT_PATH, quality=90)

print(f"Saved contact sheet: {OUTPUT_PATH}")
print(f"Images included: {len(metadata)}")
