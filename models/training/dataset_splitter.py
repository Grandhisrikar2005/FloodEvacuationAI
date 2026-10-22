import os
import random
import shutil
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# =====================================================
# PATHS
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[2]

RAW_DATASET = PROJECT_ROOT / "dataset" / "raw"
SPLIT_DATASET = PROJECT_ROOT / "dataset" / "split"

# =====================================================
# CREATE DIRECTORY STRUCTURE
# =====================================================

for split in ["train", "validation", "test"]:
    (SPLIT_DATASET / split).mkdir(parents=True, exist_ok=True)

# =====================================================
# PROCESS EACH CATEGORY
# =====================================================

categories = [
    folder.name
    for folder in RAW_DATASET.iterdir()
    if folder.is_dir()
]

print("=" * 50)
print("Starting Dataset Split")
print("=" * 50)

for category in categories:

    source_folder = RAW_DATASET / category

    images = [
        image
        for image in source_folder.iterdir()
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    train_images = images[:train_count]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    destination = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    print(f"\nCategory : {category}")
    print(f"Total Images : {total_images}")

    for split_name, image_list in destination.items():

        split_folder = SPLIT_DATASET / split_name / category
        split_folder.mkdir(parents=True, exist_ok=True)

        for image in image_list:
            shutil.copy2(
                image,
                split_folder / image.name
            )

        print(f"{split_name.capitalize():12} : {len(image_list)}")

print("\n" + "=" * 50)
print("Dataset Successfully Split!")
print(f"Location : {SPLIT_DATASET}")
print("=" * 50)