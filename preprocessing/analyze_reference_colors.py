import numpy as np
import rasterio
from collections import Counter

INPUT_TIF = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\NRSC_APSAC_Vijayawada_reference_georef.tif"
)

print("=" * 70)
print("ANALYZING APSAC REFERENCE MAP COLORS")
print("=" * 70)

with rasterio.open(INPUT_TIF) as src:
    data = src.read()

red = data[0]
green = data[1]
blue = data[2]

# Current cyan selection
candidate = (
    (green > 180)
    & (blue > 180)
    & (red < 100)
)

pixels = np.column_stack([
    red[candidate],
    green[candidate],
    blue[candidate]
])

print("\nCandidate cyan pixels:", len(pixels))

# Convert to integers for counting
pixels = np.clip(pixels, 0, 255).astype(np.uint8)

colors = Counter(map(tuple, pixels))

print("\nMost common RGB colors among candidate pixels:")
print("-" * 70)

for color, count in colors.most_common(30):

    percentage = count / len(pixels) * 100

    print(
        f"RGB {color}: "
        f"{count:,} pixels "
        f"({percentage:.2f}%)"
    )

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)