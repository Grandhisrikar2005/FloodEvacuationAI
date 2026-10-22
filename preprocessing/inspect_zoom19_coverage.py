
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "dataset" / "dataset_metadata_v3.csv"

# Existing flood-study location: Vijayawada / Krishna River
PROJECT_LAT = 16.5308984
PROJECT_LON = 80.6176758


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def main():
    if not CSV_PATH.exists():
        print(f"ERROR: Metadata file not found:\n{CSV_PATH}")
        print("Check the CSV filename and dataset folder.")
        return

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("ERROR: The metadata file contains no records.")
        return

    for row in rows:
        row["latitude"] = float(row["latitude"])
        row["longitude"] = float(row["longitude"])
        row["tile_x"] = int(row["tile_x"])
        row["tile_y"] = int(row["tile_y"])
        row["zoom"] = int(row["zoom"])

    print("=" * 65)
    print("ZOOM-19 LAND-COVER DATASET COVERAGE")
    print("=" * 65)

    print(f"Metadata records : {len(rows)}")
    print(f"Zoom levels      : {sorted({r['zoom'] for r in rows})}")
    print(
        f"Latitude range   : "
        f"{min(r['latitude'] for r in rows):.6f} to "
        f"{max(r['latitude'] for r in rows):.6f}"
    )
    print(
        f"Longitude range  : "
        f"{min(r['longitude'] for r in rows):.6f} to "
        f"{max(r['longitude'] for r in rows):.6f}"
    )

    print("\nCLASS COUNTS")
    counts = {}
    for row in rows:
        category = row["category"]
        counts[category] = counts.get(category, 0) + 1

    for category, count in sorted(counts.items()):
        print(f"{category:15s}: {count}")

    print("\nTILE GRID")
    xs = sorted({r["tile_x"] for r in rows})
    ys = sorted({r["tile_y"] for r in rows})

    print(f"Unique tile_x values : {len(xs)} ({min(xs)} to {max(xs)})")
    print(f"Unique tile_y values : {len(ys)} ({min(ys)} to {max(ys)})")
    print(f"Zoom-19 grid extent  : {len(xs)} columns x {len(ys)} rows")
    print(
        "Note: this is the bounding grid, not necessarily a fully "
        "populated continuous mosaic."
    )

    print("\n10 LOCATIONS NEAREST TO THE PROJECT CENTRE")
    nearest = sorted(
        rows,
        key=lambda r: haversine_km(
            PROJECT_LAT,
            PROJECT_LON,
            r["latitude"],
            r["longitude"],
        ),
    )

    for row in nearest[:10]:
        distance = haversine_km(
            PROJECT_LAT,
            PROJECT_LON,
            row["latitude"],
            row["longitude"],
        )

        print(
            f"{row['location'][:28]:28s} "
            f"{row['category']:12s} "
            f"distance={distance:6.2f} km "
            f"tile=({row['tile_x']},{row['tile_y']})"
        )

    print("\nUNIQUE TILE COORDINATES")
    print(
        "Unique (tile_x, tile_y, zoom) combinations:",
        len({
            (r["tile_x"], r["tile_y"], r["zoom"])
            for r in rows
        }),
    )

    print("\nInspection complete. No files were changed.")


if __name__ == "__main__":
    main()
