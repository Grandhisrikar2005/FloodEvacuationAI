import os

import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from pyproj import Transformer


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

INPUT_IMAGE = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\NRSC_APSAC_Vijayawada_map_only.png"
)

OUTPUT_TIF = (
    r"C:\Users\SRIKA\PycharmProjects\FloodEvacuationAI"
    r"\dataset\temporal\krishna_flood_2024\reference"
    r"\NRSC_APSAC_Vijayawada_reference_georef.tif"
)


# ---------------------------------------------------------
# OFFICIAL NRSC / APSAC GEOGRAPHIC CORNERS
#
# GPTS from the PDF:
#
# 16.4848 80.5068
# 16.8654 80.5082
# 16.8638 80.8419
# 16.4832 80.8396
#
# These correspond to:
# bottom-left
# top-left
# top-right
# bottom-right
# ---------------------------------------------------------

geo_corners = [
    (80.5068, 16.4848),  # bottom-left
    (80.5082, 16.8654),  # top-left
    (80.8419, 16.8638),  # top-right
    (80.8396, 16.4832),  # bottom-right
]


print("=" * 70)
print("GEORREFERENCING NRSC / APSAC FLOOD REFERENCE MAP")
print("=" * 70)


# ---------------------------------------------------------
# READ IMAGE
# ---------------------------------------------------------

image = Image.open(INPUT_IMAGE).convert("RGB")

width, height = image.size

print("\nInput image:")
print(INPUT_IMAGE)

print("\nImage dimensions:")
print("Width :", width)
print("Height:", height)


# ---------------------------------------------------------
# CONVERT WGS84 -> UTM ZONE 44N
# ---------------------------------------------------------

transformer = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32644",
    always_xy=True
)

utm_corners = []

for lon, lat in geo_corners:

    x, y = transformer.transform(lon, lat)

    utm_corners.append((x, y))

print("\nOfficial geographic corners converted to EPSG:32644:")

names = [
    "Bottom-left",
    "Top-left",
    "Top-right",
    "Bottom-right"
]

for name, (x, y) in zip(names, utm_corners):

    print(
        f"{name:12s}: "
        f"Easting = {x:.3f}, "
        f"Northing = {y:.3f}"
    )


# ---------------------------------------------------------
# CREATE GROUND CONTROL POINTS
# ---------------------------------------------------------

# PIL image coordinates:
#
# (0,0)               (width-1,0)
#       TOP
#
#
# (0,height-1)        (width-1,height-1)
#       BOTTOM
#
#
# Therefore:
# top-left     -> geo_corners[1]
# top-right    -> geo_corners[2]
# bottom-right -> geo_corners[3]
# bottom-left  -> geo_corners[0]

gcps = [

    GroundControlPoint(
        row=0,
        col=0,
        x=utm_corners[1][0],
        y=utm_corners[1][1]
    ),

    GroundControlPoint(
        row=0,
        col=width - 1,
        x=utm_corners[2][0],
        y=utm_corners[2][1]
    ),

    GroundControlPoint(
        row=height - 1,
        col=width - 1,
        x=utm_corners[3][0],
        y=utm_corners[3][1]
    ),

    GroundControlPoint(
        row=height - 1,
        col=0,
        x=utm_corners[0][0],
        y=utm_corners[0][1]
    ),
]


# ---------------------------------------------------------
# CALCULATE AFFINE TRANSFORM
# ---------------------------------------------------------

transform = from_gcps(gcps)

print("\nCalculated affine transform:")
print(transform)


# ---------------------------------------------------------
# WRITE GEOTIFF
# ---------------------------------------------------------

array = np.array(image)

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=3,
    dtype=array.dtype,
    crs="EPSG:32644",
    transform=transform,
    compress="deflate"
) as dst:

    dst.write(array[:, :, 0], 1)
    dst.write(array[:, :, 1], 2)
    dst.write(array[:, :, 2], 3)


# ---------------------------------------------------------
# VERIFY
# ---------------------------------------------------------

with rasterio.open(OUTPUT_TIF) as src:

    print("\n" + "=" * 70)
    print("GEOTIFF CREATED")
    print("=" * 70)

    print("\nOutput:")
    print(OUTPUT_TIF)

    print("\nCRS:")
    print(src.crs)

    print("\nDimensions:")
    print(src.width, "x", src.height)

    print("\nResolution:")
    print(src.res)

    print("\nBounds:")
    print(src.bounds)

    print("\nTransform:")
    print(src.transform)


# ---------------------------------------------------------
# CHECK PROJECT LOCATION
# ---------------------------------------------------------

project_lon = 80.6176758
project_lat = 16.5308984

project_x, project_y = transformer.transform(
    project_lon,
    project_lat
)

print("\nProject coordinate:")
print("Latitude :", project_lat)
print("Longitude:", project_lon)

print("\nEPSG:32644:")
print("Easting :", project_x)
print("Northing:", project_y)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)