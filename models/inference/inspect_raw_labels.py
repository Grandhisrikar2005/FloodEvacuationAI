
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "dataset" / "raw"
OUTPUT = ROOT / "dataset" / "raw_label_inspection.jpg"

CLASSES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

# Five samples per class, spaced through the sorted filenames.
CELL_W = 180
IMAGE_H = 150
LABEL_H = 28
COLS = 5
ROWS = len(CLASSES)

sheet = Image.new(
    "RGB",
    (COLS * CELL_W, ROWS * (IMAGE_H + LABEL_H)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for row, class_name in enumerate(CLASSES):
    folder = RAW / class_name
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not files:
        print(f"No images found for {class_name}")
        continue

    # Select five images spread across the folder.
    indices = [
        round(i * (len(files) - 1) / 4)
        for i in range(5)
    ]

    for col, index in enumerate(indices):
        path = files[index]

        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((CELL_W - 8, IMAGE_H - 8))

        x = col * CELL_W
        y = row * (IMAGE_H + LABEL_H)

        sheet.paste(image, (x + 4, y + 4))
        draw.text(
            (x + 4, y + IMAGE_H),
            f"{class_name}: {path.name}",
            fill="black"
        )

sheet.save(OUTPUT, quality=95)
print(f"Inspection sheet saved to: {OUTPUT}")
