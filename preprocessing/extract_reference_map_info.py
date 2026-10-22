import pymupdf
from PIL import Image
import io
import os

PDF_PATH = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\NRSC_APSAC_flood_reference_2024-09-01.pdf"
)

OUTPUT_DIR = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "NRSC_APSAC_Vijayawada_map_only.png"
)

print("=" * 70)
print("RECONSTRUCTING NRSC / APSAC MAP")
print("=" * 70)

doc = pymupdf.open(PDF_PATH)
page = doc[1]

images = page.get_images(full=True)

# The first 22 images are the horizontal map strips.
map_images = images[:22]

print("\nMap strips found:", len(map_images))

pieces = []

for i, img in enumerate(map_images):

    xref = img[0]

    data = doc.extract_image(xref)["image"]

    image = Image.open(io.BytesIO(data)).convert("RGB")

    print(
        f"Strip {i + 1:02d}: "
        f"{image.width} x {image.height}"
    )

    pieces.append(image)

# Verify widths
widths = [img.width for img in pieces]

if len(set(widths)) != 1:
    raise RuntimeError("Map strips do not have matching widths.")

total_height = sum(img.height for img in pieces)

print("\nFinal map dimensions:")
print("Width :", widths[0])
print("Height:", total_height)

# Create stitched map
stitched = Image.new(
    "RGB",
    (widths[0], total_height),
    "white"
)

y = 0

for image in pieces:

    stitched.paste(image, (0, y))

    y += image.height

stitched.save(
    OUTPUT_PATH,
    format="PNG"
)

print("\n" + "=" * 70)
print("MAP RECONSTRUCTION COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_PATH)

print(
    f"\nFinal image size: "
    f"{stitched.width} x {stitched.height}"
)

print("=" * 70)