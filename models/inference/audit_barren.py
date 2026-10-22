
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "dataset" / "raw" / "Barren"
OUTPUT = ROOT / "dataset" / "barren_audit.jpg"

files = sorted(
    p for p in FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

cols = 5
cell_w, cell_h = 200, 175
rows = (len(files) + cols - 1) // cols

sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    with Image.open(path) as im:
        image = im.convert("RGB")
        image.thumbnail((cell_w - 8, cell_h - 32))

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(image, (x + 4, y + 4))
    draw.text((x + 4, y + cell_h - 24), path.name, fill="black")

sheet.save(OUTPUT, quality=92)
print(f"Images inspected: {len(files)}")
print(f"Audit sheet saved to: {OUTPUT}")
