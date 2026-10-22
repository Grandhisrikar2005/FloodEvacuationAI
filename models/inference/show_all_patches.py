
from pathlib import Path
from PIL import Image, ImageDraw

# Folder containing the 9 diagnostic patches
patch_dir = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "zoom19_pilot"
    / "diagnostic_patches"
)

patch_names = [
    "top_left", "top_centre", "top_right",
    "middle_left", "centre", "middle_right",
    "bottom_left", "bottom_centre", "bottom_right"
]

patch_size = (280, 280)
label_height = 35
gap = 12

sheet_width = 3 * patch_size[0] + 4 * gap
sheet_height = 3 * (patch_size[1] + label_height) + 4 * gap

sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
draw = ImageDraw.Draw(sheet)

for i, name in enumerate(patch_names):
    image_path = patch_dir / f"{name}.jpg"

    if not image_path.exists():
        print(f"Missing file: {image_path}")
        continue

    image = Image.open(image_path).convert("RGB")
    image = image.resize(patch_size)

    row = i // 3
    col = i % 3

    x = gap + col * (patch_size[0] + gap)
    y = gap + row * (patch_size[1] + label_height + gap)

    sheet.paste(image, (x, y))
    draw.text(
        (x + 5, y + patch_size[1] + 5),
        name.replace("_", " ").upper(),
        fill="black"
    )

output_path = patch_dir / "contact_sheet.jpg"
sheet.save(output_path, quality=95)

print(f"Contact sheet saved to:\n{output_path}")
