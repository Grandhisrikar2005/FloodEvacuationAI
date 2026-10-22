
from pathlib import Path
from PIL import Image, ImageDraw

project_root = Path(__file__).resolve().parents[2]
train_dir = project_root / "dataset" / "split" / "train"
patch_dir = project_root / "dataset" / "zoom19_pilot" / "diagnostic_patches"

classes = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

patch_names = [
    "top_left",
    "top_centre",
    "top_right",
    "middle_left",
    "centre",
    "middle_right",
    "bottom_left",
    "bottom_centre",
    "bottom_right",
]

cell_w, cell_h = 240, 270
cols = 3
rows = len(classes) + 3

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)
draw = ImageDraw.Draw(sheet)

# First, show one original training image from each class.
for i, class_name in enumerate(classes):
    folder = train_dir / class_name
    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        print(f"No images found for {class_name}")
        continue

    image = Image.open(images[0]).convert("RGB")
    image.thumbnail((cell_w - 16, cell_h - 40))

    x = (i % cols) * cell_w
    y = (i // cols) * cell_h

    sheet.paste(image, (x + 8, y + 30))
    draw.text((x + 8, y + 8), f"TRAIN: {class_name}", fill="black")

# Then show all nine new imagery patches.
for i, name in enumerate(patch_names):
    image_path = patch_dir / f"{name}.jpg"
    if not image_path.exists():
        print(f"Missing patch: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((cell_w - 16, cell_h - 40))

    x = (i % cols) * cell_w
    y = (len(classes) + i // cols) * cell_h

    sheet.paste(image, (x + 8, y + 30))
    draw.text((x + 8, y + 8), f"PILOT: {name}", fill="black")

output = project_root / "dataset" / "zoom19_pilot" / "training_comparison.jpg"
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output, quality=95)

print(f"Comparison saved to: {output}")
