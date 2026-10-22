import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# FILE PATHS
# ============================================================

REFERENCE_MASK = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\reference_flood_mask_2024-09-01.tif"
)

CHANGE_MAP = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\flood_analysis"
    r"\sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

OUTPUT_IMAGE = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\zoomed_reference_vs_sar_overlay.png"
)


# ============================================================
# SETTINGS
# ============================================================

SAR_THRESHOLD = -3.0

# Extra margin around APSAC map
MARGIN_PIXELS = 100

# Maximum display size
MAX_SIZE = 1600


print("=" * 70)
print("ZOOMED APSAC REFERENCE VS SENTINEL-1 FLOOD OVERLAY")
print("=" * 70)


# ============================================================
# READ SAR CHANGE MAP
# ============================================================

with rasterio.open(CHANGE_MAP) as sar_src:

    sar_change = sar_src.read(1)

    sar_transform = sar_src.transform
    sar_crs = sar_src.crs

    sar_width = sar_src.width
    sar_height = sar_src.height

    sar_nodata = sar_src.nodata

    print("\nSAR dimensions:")
    print(f"{sar_width} x {sar_height}")

    print("SAR CRS:")
    print(sar_crs)

    print("SAR resolution:")
    print(sar_src.res)


# ============================================================
# READ APSAC REFERENCE MASK
# ============================================================

with rasterio.open(REFERENCE_MASK) as ref_src:

    reference_mask = ref_src.read(1)

    ref_transform = ref_src.transform
    ref_crs = ref_src.crs

    ref_width = ref_src.width
    ref_height = ref_src.height

    print("\nAPSAC dimensions:")
    print(f"{ref_width} x {ref_height}")

    print("APSAC CRS:")
    print(ref_crs)

    print("APSAC resolution:")
    print(ref_src.res)


# ============================================================
# SAR VALID PIXELS
# ============================================================

if sar_nodata is not None:

    valid_sar = (
        np.isfinite(sar_change)
        & (sar_change != sar_nodata)
    )

else:

    valid_sar = np.isfinite(sar_change)


# ============================================================
# SENTINEL-1 FLOOD MASK
# ============================================================

sar_flood = (
    valid_sar
    & (sar_change <= SAR_THRESHOLD)
)

print("\nSentinel-1 threshold:")
print(f"{SAR_THRESHOLD} dB")

print(
    "Sentinel-1 flood pixels:",
    int(np.sum(sar_flood))
)


# ============================================================
# REPROJECT APSAC MASK TO SAR GRID
# ============================================================

print("\nReprojecting APSAC reference onto SAR grid...")


reference_on_sar = np.zeros(
    (sar_height, sar_width),
    dtype=np.uint8
)


reference_coverage = np.ones(
    (ref_height, ref_width),
    dtype=np.uint8
)


reference_valid_on_sar = np.zeros(
    (sar_height, sar_width),
    dtype=np.uint8
)


# ------------------------------------------------------------
# Reproject flood mask
# ------------------------------------------------------------

reproject(
    source=reference_mask.astype(np.uint8),
    destination=reference_on_sar,

    src_transform=ref_transform,
    src_crs=ref_crs,

    dst_transform=sar_transform,
    dst_crs=sar_crs,

    resampling=Resampling.nearest
)


# ------------------------------------------------------------
# Reproject reference coverage
# ------------------------------------------------------------

reproject(
    source=reference_coverage,
    destination=reference_valid_on_sar,

    src_transform=ref_transform,
    src_crs=ref_crs,

    dst_transform=sar_transform,
    dst_crs=sar_crs,

    resampling=Resampling.nearest
)


reference_on_sar = reference_on_sar > 0

reference_valid_on_sar = (
    reference_valid_on_sar > 0
)


print(
    "APSAC flood pixels on SAR grid:",
    int(np.sum(reference_on_sar))
)


# ============================================================
# FIND APSAC AREA ON SAR GRID
# ============================================================

rows, cols = np.where(reference_valid_on_sar)


if len(rows) == 0:

    raise RuntimeError(
        "Could not find the APSAC reference area on the SAR grid."
    )


min_row = max(
    0,
    rows.min() - MARGIN_PIXELS
)

max_row = min(
    sar_height,
    rows.max() + MARGIN_PIXELS + 1
)

min_col = max(
    0,
    cols.min() - MARGIN_PIXELS
)

max_col = min(
    sar_width,
    cols.max() + MARGIN_PIXELS + 1
)


print("\nAPSAC area located inside SAR scene:")

print(
    f"Rows: {min_row} to {max_row}"
)

print(
    f"Columns: {min_col} to {max_col}"
)


# ============================================================
# CROP BOTH DATASETS
# ============================================================

sar_flood_crop = sar_flood[
    min_row:max_row,
    min_col:max_col
]

reference_crop = reference_on_sar[
    min_row:max_row,
    min_col:max_col
]

valid_crop = common_valid = (
    valid_sar[
        min_row:max_row,
        min_col:max_col
    ]
    &
    reference_valid_on_sar[
        min_row:max_row,
        min_col:max_col
    ]
)


crop_height, crop_width = sar_flood_crop.shape


print("\nZoomed area dimensions:")
print(
    f"{crop_width} x {crop_height}"
)


# ============================================================
# DOWNSAMPLE ONLY IF NECESSARY
# ============================================================

scale = min(
    1.0,
    MAX_SIZE / crop_height,
    MAX_SIZE / crop_width
)

display_height = max(
    1,
    int(crop_height * scale)
)

display_width = max(
    1,
    int(crop_width * scale)
)


# ============================================================
# SIMPLE NEAREST-NEIGHBOR DOWNSAMPLING
# ============================================================

def downsample(array, new_height, new_width):

    old_height, old_width = array.shape

    row_indices = np.linspace(
        0,
        old_height - 1,
        new_height
    ).astype(np.int32)

    col_indices = np.linspace(
        0,
        old_width - 1,
        new_width
    ).astype(np.int32)

    return array[
        row_indices[:, None],
        col_indices[None, :]
    ]


sar_small = downsample(
    sar_flood_crop.astype(np.uint8),
    display_height,
    display_width
)


reference_small = downsample(
    reference_crop.astype(np.uint8),
    display_height,
    display_width
)


valid_small = downsample(
    valid_crop.astype(np.uint8),
    display_height,
    display_width
)


# ============================================================
# CREATE RGB OVERLAY
# ============================================================

overlay = np.zeros(
    (
        display_height,
        display_width,
        3
    ),
    dtype=np.uint8
)


# ============================================================
# APSAC ONLY = RED
# ============================================================

apsac_only = (
    (reference_small == 1)
    & (sar_small == 0)
)

overlay[apsac_only] = [
    255,
    0,
    0
]


# ============================================================
# SENTINEL-1 ONLY = BLUE
# ============================================================

sar_only = (
    (reference_small == 0)
    & (sar_small == 1)
)

overlay[sar_only] = [
    0,
    0,
    255
]


# ============================================================
# BOTH = MAGENTA
# ============================================================

both = (
    (reference_small == 1)
    & (sar_small == 1)
)

overlay[both] = [
    255,
    0,
    255
]


# ============================================================
# NON-FLOOD VALID AREA = GRAY
# ============================================================

normal_area = (
    (reference_small == 0)
    & (sar_small == 0)
    & (valid_small == 1)
)

overlay[normal_area] = [
    60,
    60,
    60
]


# ============================================================
# OUTSIDE VALID AREA = BLACK
# ============================================================

outside = (
    valid_small == 0
)

overlay[outside] = [
    0,
    0,
    0
]


# ============================================================
# STATISTICS
# ============================================================

apsac_only_count = int(
    np.sum(apsac_only)
)

sar_only_count = int(
    np.sum(sar_only)
)

both_count = int(
    np.sum(both)
)


print("\nZoomed overlay statistics:")
print("-" * 50)

print(
    "APSAC only:",
    apsac_only_count
)

print(
    "Sentinel-1 only:",
    sar_only_count
)

print(
    "Both:",
    both_count
)


# ============================================================
# SAVE IMAGE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_IMAGE),
    exist_ok=True
)


plt.imsave(
    OUTPUT_IMAGE,
    overlay
)


print("\nZoomed overlay saved to:")
print(OUTPUT_IMAGE)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(14, 10)
)

plt.imshow(overlay)

plt.title(
    "Zoomed APSAC Reference vs Sentinel-1 Flood\n"
    f"Sentinel-1 threshold = {SAR_THRESHOLD} dB"
)

plt.axis("off")


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Patch(
        facecolor="red",
        label="APSAC reference only"
    ),

    Patch(
        facecolor="blue",
        label="Sentinel-1 only"
    ),

    Patch(
        facecolor="magenta",
        label="Both APSAC + Sentinel-1"
    ),

    Patch(
        facecolor="gray",
        label="Non-flood valid area"
    ),

]


plt.legend(
    handles=legend_elements,
    loc="lower right"
)

plt.tight_layout()

plt.show()


print("\n" + "=" * 70)
print("ZOOMED OVERLAY COMPLETE")
print("=" * 70)