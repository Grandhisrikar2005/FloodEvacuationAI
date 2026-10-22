import rasterio
import matplotlib.pyplot as plt


INPUT_TIF = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\reference_flood_mask_2024-09-01.tif"
)

with rasterio.open(INPUT_TIF) as src:
    mask = src.read(1)

plt.figure(figsize=(10, 12))

plt.imshow(mask, cmap="gray")

plt.title(
    "NRSC / APSAC Official Flood Reference Mask\n"
    "1 September 2024 - Vijayawada"
)

plt.axis("off")
plt.tight_layout()

plt.show()