
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "dataset" / "barren_label_review.csv"
IMAGE_DIR = ROOT / "dataset" / "barren_v2" / "Barren"
OUTPUT_DIR = ROOT / "dataset" / "barren_reviewed_v1"

CLASSES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

df = pd.read_csv(CSV_PATH, keep_default_na=False)

required = {"filename", "verified_label", "review_status"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain: {required}")

# Only use images explicitly reviewed and assigned a valid label.
verified = df[
    (df["review_status"].str.strip().str.lower() == "reviewed")
    & (df["verified_label"].str.strip().isin(CLASSES))
].copy()

if verified.empty:
    print("No verified images found. Review and save some labels first.")
    print("\nCurrent review status:")
    print(df["review_status"].value_counts().to_string())
    raise SystemExit(0)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
missing = []

for class_name in CLASSES:
    (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

for _, row in verified.iterrows():
    filename = Path(str(row["filename"])).name
    source = IMAGE_DIR / filename
    label = row["verified_label"].strip()

    if not source.is_file():
        missing.append(filename)
        continue

    destination = OUTPUT_DIR / label / filename
    shutil.copy2(source, destination)
    copied += 1

print(f"\nVerified images copied: {copied}")
print(f"Missing image files: {len(missing)}")
print(f"Output folder: {OUTPUT_DIR}")

print("\nVerified image counts by class:")
for class_name in CLASSES:
    count = len(list((OUTPUT_DIR / class_name).glob("*")))
    print(f"{class_name}: {count}")

print("\nNOT INCLUDED:")
print("Unreviewed images, uncertain images, and invalid labels.")
print("This is a verified subset, not yet a complete training dataset.")
if missing:
    print("\nMissing files:")
    print("\n".join(missing))
