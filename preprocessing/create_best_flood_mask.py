
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage


# ============================================================
# PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE = (
    PROJECT_ROOT
    / "dataset"
    / "temporal"
    / "krishna_flood_2024"
    / "flood_analysis"
)

CHANGE_MAP = (
    BASE / "sar_vv_change_2024-08-20_to_2024-09-01.tif"
)

OUTPUT_MASK = (
    BASE / "best_flood_mask_v3_2024-09-01.tif"
)

TEMP_LABELS = BASE / "flood_components_v3_temp.tif"

THRESHOLD_DB = -2.0
MIN_COMPONENT_PIXELS = 1000
NODATA_VALUE = 255
WINDOW_SIZE = 1024

STRUCTURE = np.ones((3, 3), dtype=np.uint8)


# ============================================================
# PASS 1: LABEL EACH WINDOW AND WRITE UNIQUE LABELS
# ============================================================

def label_windows():
    total_valid = 0
    next_label = 1

    with rasterio.open(CHANGE_MAP) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="int32",
            count=1,
            nodata=0,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(TEMP_LABELS, "w", **profile) as dst:
            for row in range(0, src.height, WINDOW_SIZE):
                for col in range(0, src.width, WINDOW_SIZE):
                    window = Window(
                        col,
                        row,
                        min(WINDOW_SIZE, src.width - col),
                        min(WINDOW_SIZE, src.height - row)
                    )

                    change = src.read(1, window=window)
                    valid = np.isfinite(change)

                    if src.nodata is not None:
                        valid &= change != src.nodata

                    candidates = valid & (change <= THRESHOLD_DB)

                    local_labels, count = ndimage.label(
                        candidates,
                        structure=STRUCTURE
                    )

                    labels = np.zeros(
                        candidates.shape, dtype=np.int32
                    )

                    if count:
                        labels[local_labels > 0] = (
                            local_labels[local_labels > 0]
                            + next_label - 1
                        )
                        next_label += count

                    dst.write(labels, 1, window=window)
                    total_valid += int(valid.sum())

                print(
                    f"Pass 1: rows "
                    f"{min(row + WINDOW_SIZE, src.height)}/{src.height}"
                )

    return total_valid, next_label - 1


# ============================================================
# PASS 2: MERGE LABELS ACROSS WINDOW BOUNDARIES
# ============================================================

class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size + 1, dtype=np.int64)
        self.rank = np.zeros(size + 1, dtype=np.uint8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def merge_boundary_labels(label_count):
    uf = UnionFind(label_count)

    with rasterio.open(TEMP_LABELS) as src:
        width, height = src.width, src.height

        # Merge across vertical window boundaries.
        for col in range(WINDOW_SIZE, width, WINDOW_SIZE):
            left = src.read(
                1,
                window=Window(
                    col - 1, 0, 1, height
                )
            )[:, 0]

            right = src.read(
                1,
                window=Window(
                    col, 0, 1, height
                )
            )[:, 0]

            for offset in (-1, 0, 1):
                if offset < 0:
                    a, b = left[1:], right[:-1]
                elif offset > 0:
                    a, b = left[:-1], right[1:]
                else:
                    a, b = left, right

                pairs = np.column_stack((a, b))
                pairs = pairs[
                    (pairs[:, 0] > 0)
                    & (pairs[:, 1] > 0)
                    & (pairs[:, 0] != pairs[:, 1])
                ]

                if pairs.size:
                    for x, y in np.unique(pairs, axis=0):
                        uf.union(int(x), int(y))

            print(f"Merged vertical boundary at column {col}")

        # Merge across horizontal window boundaries.
        for row in range(WINDOW_SIZE, height, WINDOW_SIZE):
            above = src.read(
                1,
                window=Window(
                    0, row - 1, width, 1
                )
            )[0, :]

            below = src.read(
                1,
                window=Window(
                    0, row, width, 1
                )
            )[0, :]

            for offset in (-1, 0, 1):
                if offset < 0:
                    a, b = above[1:], below[:-1]
                elif offset > 0:
                    a, b = above[:-1], below[1:]
                else:
                    a, b = above, below

                pairs = np.column_stack((a, b))
                pairs = pairs[
                    (pairs[:, 0] > 0)
                    & (pairs[:, 1] > 0)
                    & (pairs[:, 0] != pairs[:, 1])
                ]

                if pairs.size:
                    for x, y in np.unique(pairs, axis=0):
                        uf.union(int(x), int(y))

            print(f"Merged horizontal boundary at row {row}")

    # Compact every label to its global component root.
    roots = np.arange(label_count + 1, dtype=np.int64)
    for label in range(1, label_count + 1):
        roots[label] = uf.find(label)

    unique_roots, inverse = np.unique(
        roots[1:], return_inverse=True
    )

    # Maps original labels to compact global component IDs.
    compact = np.zeros(label_count + 1, dtype=np.int32)
    compact[1:] = inverse + 1

    return compact, len(unique_roots)


# ============================================================
# PASS 3: COUNT GLOBAL COMPONENT SIZES
# ============================================================

def count_global_components(compact):
    counts = np.zeros(int(compact.max()) + 1, dtype=np.int64)

    with rasterio.open(TEMP_LABELS) as src:
        for row in range(0, src.height, WINDOW_SIZE):
            for col in range(0, src.width, WINDOW_SIZE):
                window = Window(
                    col,
                    row,
                    min(WINDOW_SIZE, src.width - col),
                    min(WINDOW_SIZE, src.height - row)
                )

                labels = src.read(1, window=window)
                positive = labels > 0

                if positive.any():
                    global_ids = compact[labels[positive]]
                    counts += np.bincount(
                        global_ids,
                        minlength=len(counts)
                    )

            print(
                f"Pass 3: rows "
                f"{min(row + WINDOW_SIZE, src.height)}/{src.height}"
            )

    return counts


# ============================================================
# PASS 4: WRITE THE FINAL CANDIDATE MASK
# ============================================================

def write_final_mask(compact, counts):
    total_filtered = 0
    total_nodata = 0
    total_valid = 0
    total_raw = 0

    with rasterio.open(CHANGE_MAP) as src:
        profile = src.profile.copy()
        profile.update(
            dtype="uint8",
            count=1,
            nodata=NODATA_VALUE,
            compress="lzw",
            BIGTIFF="IF_SAFER"
        )

        with rasterio.open(OUTPUT_MASK, "w", **profile) as dst:
            for row in range(0, src.height, WINDOW_SIZE):
                for col in range(0, src.width, WINDOW_SIZE):
                    window = Window(
                        col,
                        row,
                        min(WINDOW_SIZE, src.width - col),
                        min(WINDOW_SIZE, src.height - row)
                    )

                    change = src.read(1, window=window)
                    valid = np.isfinite(change)

                    if src.nodata is not None:
                        valid &= change != src.nodata

                    raw = valid & (change <= THRESHOLD_DB)

                    with rasterio.open(TEMP_LABELS) as label_src:
                        labels = label_src.read(1, window=window)

                    output = np.zeros(change.shape, dtype=np.uint8)

                    has_labels = labels > 0
                    global_ids = np.zeros(
                        labels.shape, dtype=np.int32
                    )
                    global_ids[has_labels] = compact[
                        labels[has_labels]
                    ]

                    keep = np.zeros(labels.shape, dtype=bool)
                    keep[has_labels] = (
                        counts[global_ids[has_labels]]
                        >= MIN_COMPONENT_PIXELS
                    )

                    output[keep & valid] = 1
                    output[~valid] = NODATA_VALUE

                    dst.write(output, 1, window=window)

                    total_valid += int(valid.sum())
                    total_raw += int(raw.sum())
                    total_filtered += int((output == 1).sum())
                    total_nodata += int((~valid).sum())

                print(
                    f"Pass 4: rows "
                    f"{min(row + WINDOW_SIZE, src.height)}/{src.height}"
                )

    return total_valid, total_raw, total_filtered, total_nodata


# ============================================================
# MAIN
# ============================================================

def main():
    if not CHANGE_MAP.exists():
        raise FileNotFoundError(CHANGE_MAP)

    if OUTPUT_MASK.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {OUTPUT_MASK}"
        )

    print("=" * 70)
    print("GLOBAL CONNECTED-COMPONENT FLOOD CANDIDATE MASK")
    print("=" * 70)
    print("Threshold:", THRESHOLD_DB, "dB")
    print("Minimum component size:", MIN_COMPONENT_PIXELS)
    print("Output:", OUTPUT_MASK)

    try:
        valid_count, label_count = label_windows()
        print("Valid input pixels:", f"{valid_count:,}")
        print("Local component labels:", f"{label_count:,}")

        compact, global_count = merge_boundary_labels(label_count)
        print("Global components after merging:", f"{global_count:,}")

        counts = count_global_components(compact)

        stats = write_final_mask(compact, counts)

        print("\n" + "=" * 70)
        print("MASK V3 COMPLETE")
        print("=" * 70)
        print("Valid pixels:", f"{stats[0]:,}")
        print("Raw candidates:", f"{stats[1]:,}")
        print("Filtered candidates:", f"{stats[2]:,}")
        print("NoData pixels:", f"{stats[3]:,}")
        print("Output:", OUTPUT_MASK)
        print("Encoding: 0=non-candidate, 1=candidate, 255=NoData")

        with rasterio.open(OUTPUT_MASK) as result:
            print("Output CRS:", result.crs)
            print("Output NoData:", result.nodata)

    finally:
        if TEMP_LABELS.exists():
            TEMP_LABELS.unlink()
            print("Temporary label raster removed.")


if __name__ == "__main__":
    main()
