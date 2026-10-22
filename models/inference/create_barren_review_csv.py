
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[2]
metadata_path = root / "dataset" / "barren_metadata_v2.csv"
output_path = root / "dataset" / "barren_label_review.csv"

df = pd.read_csv(metadata_path)

df["original_label"] = df["category"]
df["verified_label"] = ""
df["review_status"] = "pending"
df["review_notes"] = ""

df.to_csv(output_path, index=False)

print(f"Created: {output_path}")
print(f"Images to review: {len(df)}")
print("\nReview labels: Agriculture, Barren, Roads, Urban, Vegetation, Water")
