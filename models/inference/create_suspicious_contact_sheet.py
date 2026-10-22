
from pathlib import Path
import csv
import math
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "evaluation" / "suspicious_labels.csv"
OUTPUT_PATH = ROOT / "outputs" / "evaluation" / "suspicious_contact_sheet.jpg"

with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("No suspicious candidates found. No contact sheet created.")
    raise SystemExit

columns = 3
cell_w, cell_h = 360, 310
rows_count = math.ceil(len(rows) / columns)

sheet = Image.new(
    "RGB",
    (columns * cell_w, rows_count * cell_h),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, row in enumerate(rows):
    image_path = ROOT / row["path"]
    x = (i % columns) * cell_w
    y = (i // columns) * cell_h

    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((cell_w - 20, 220))
            sheet.paste(im, (x + 10, y + 10))

        confidence = float(row["prediction_confidence"]) * 100
        text = (
            f'{row["filename"]}\n'
            f'Actual folder: {row["folder_label"]}\n'
            f'Prediction: {row["model_prediction"]} ({confidence:.1f}%)'
        )
        draw.multiline_text(
            (x + 10, y + 240),
            text,
            fill="black",
            spacing=5
        )
    except Exception as exc:
        draw.text((x + 10, y + 10), f'Error: {exc}', fill="red")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT_PATH, quality=95)

print(f"Contact sheet created: {OUTPUT_PATH}")
