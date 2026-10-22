
from pathlib import Path
from collections import defaultdict
import hashlib
import random
import shutil

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    ROOT / "dataset" / "raw",
    ROOT / "dataset" / "Final",
]

OUTPUT = ROOT / "dataset" / "expanded_v1"

CLASSES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

# Do not overwrite an existing experiment.
if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise FileExistsError(
        f"{OUTPUT} already contains files. "
        "Choose a new output folder before running again."
    )

# --------------------------------------------------
# Collect files and remove exact duplicates
# --------------------------------------------------
records = []
seen_hashes = set()

for source in SOURCES:
    for class_name in CLASSES:
        class_dir = source / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing class folder: {class_dir}")
            continue

        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if digest in seen_hashes:
                continue

            seen_hashes.add(digest)
            records.append((class_name, path))

# --------------------------------------------------
# Split each class independently
# --------------------------------------------------
random.seed(SEED)

splits = {
    "train": [],
    "validation": [],
    "test": [],
}

for class_name in CLASSES:
    class_records = [
        record for record in records if record[0] == class_name
    ]
    random.shuffle(class_records)

    n = len(class_records)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    splits["train"].extend(class_records[:n_train])
    splits["validation"].extend(
        class_records[n_train:n_train + n_val]
    )
    splits["test"].extend(class_records[n_train + n_val:])

# --------------------------------------------------
# Copy files into a separate dataset
# --------------------------------------------------
for split_name, split_records in splits.items():
    for index, (class_name, source_path) in enumerate(split_records):
        destination_dir = OUTPUT / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        destination = destination_dir / (
            f"{class_name.lower()}_{index:04d}{source_path.suffix.lower()}"
        )
        shutil.copy2(source_path, destination)

# --------------------------------------------------
# Report results
# --------------------------------------------------
print("\nEXPANDED DATASET CREATED")
print(f"Output: {OUTPUT}")
print(f"Unique exact file contents: {len(records)}")

for split_name in ("train", "validation", "test"):
    print(f"\n{split_name.upper()}")
    for class_name in CLASSES:
        count = len(list((OUTPUT / split_name / class_name).glob("*")))
        print(f"  {class_name}: {count}")

print(
    "\nImportant: this split removes exact duplicate files only. "
    "Near-duplicate images may still cross splits; verify labels and "
    "image provenance before treating test results as independent."
)
