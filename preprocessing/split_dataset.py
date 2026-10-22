import shutil
import random
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "final"
OUTPUT_PATH = BASE_DIR / "dataset" / "split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# =====================================================
# CREATE OUTPUT FOLDERS
# =====================================================

if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)

for split in ["train", "validation", "test"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

# =====================================================
# PROCESS EACH CLASS
# =====================================================

for class_folder in DATASET_PATH.iterdir():

    if not class_folder.is_dir():
        continue

    images = list(class_folder.glob("*"))

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for split_name, split_images in [
        ("train", train_images),
        ("validation", val_images),
        ("test", test_images),
    ]:

        destination = OUTPUT_PATH / split_name / class_folder.name
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy(image, destination / image.name)

# =====================================================
# DONE
# =====================================================

print("=" * 50)
print("Dataset Split Completed Successfully")
print("=" * 50)