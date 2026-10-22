from pathlib import Path
import sys

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================================================
# IMPORT CONSTANTS
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

UTILS_PATH = CURRENT_FILE.parents[1] / "utils"

sys.path.append(str(UTILS_PATH))

from constants import (
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    CLASS_NAMES
)

# =====================================================
# IMAGE TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# DATASETS
# =====================================================

train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    VALIDATION_PATH,
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=test_transform
)

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"Training Images    : {len(train_dataset)}")
    print(f"Validation Images  : {len(validation_dataset)}")
    print(f"Testing Images     : {len(test_dataset)}")

    print("\nClasses:")

    for index, class_name in enumerate(train_dataset.classes):
        print(f"{index} -> {class_name}")

    print("=" * 50)