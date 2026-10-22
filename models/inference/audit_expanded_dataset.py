
from pathlib import Path
from collections import Counter, defaultdict
from hashlib import sha256
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "expanded_v1"
OUTPUT = ROOT / "outputs" / "evaluation" / "dataset_audit.csv"

SPLITS = ["train", "validation", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

records = []
hash_locations = defaultdict(list)
bad_files = []

for split in SPLITS:
    split_dir = DATASET / split
    if not split_dir.exists():
        print(f"WARNING: Missing split folder: {split_dir}")
        continue

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    image.verify()

                digest = sha256(path.read_bytes()).hexdigest()
                hash_locations[digest].append((split, class_dir.name, str(path)))
                records.append({
                    "split": split,
                    "label": class_dir.name,
                    "filename": path.name,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": digest,
                    "readable": True,
                })
            except Exception as exc:
                bad_files.append((str(path), str(exc)))
                records.append({
                    "split": split,
                    "label": class_dir.name,
                    "filename": path.name,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": "",
                    "readable": False,
                })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "split", "label", "filename", "path",
            "sha256", "readable"
        ],
    )
    writer.writeheader()
    writer.writerows(records)

print("\n===== DATASET AUDIT =====")
print(f"Images inspected: {len(records)}")
print(f"Unreadable images: {len(bad_files)}")

print("\nClass counts:")
counts = Counter((r["split"], r["label"]) for r in records)
for split in SPLITS:
    print(f"\n{split}:")
    for label in sorted({r["label"] for r in records if r["split"] == split}):
        print(f"  {label}: {counts[(split, label)]}")

duplicates = [
    locations for locations in hash_locations.values()
    if len(locations) > 1
]

cross_split = [
    locations for locations in duplicates
    if len({item[0] for item in locations}) > 1
]

conflicting_labels = [
    locations for locations in duplicates
    if len({item[1] for item in locations}) > 1
]

print(f"\nExact duplicate groups: {len(duplicates)}")
print(f"Duplicate groups crossing splits: {len(cross_split)}")
print(f"Duplicate groups with conflicting labels: {len(conflicting_labels)}")

if bad_files:
    print("\nUnreadable files:")
    for path, error in bad_files:
        print(f"  {path}: {error}")

print(f"\nDetailed report saved to: {OUTPUT}")
print("No dataset files or model checkpoints were modified.")
