
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# PATHS AND SETTINGS
# =====================================================
ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = ROOT / "dataset" / "expanded_v1"
MODEL_OUTPUT = ROOT / "models" / "saved_models"
MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_OUTPUT / "expanded_best_model.pth"
CLASS_NAMES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
SEED = 42

# =====================================================
# REPRODUCIBILITY
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Dataset: {DATASET_ROOT}")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

# =====================================================
# TRANSFORMS
# =====================================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train", transform=train_transform
)
validation_dataset = datasets.ImageFolder(
    DATASET_ROOT / "validation", transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    DATASET_ROOT / "test", transform=eval_transform
)

expected_classes = sorted(CLASS_NAMES)
for name, dataset in [
    ("train", train_dataset),
    ("validation", validation_dataset),
    ("test", test_dataset),
]:
    if dataset.classes != expected_classes:
        raise ValueError(
            f"{name} class order mismatch: {dataset.classes}"
        )

if not train_dataset or not validation_dataset or not test_dataset:
    raise ValueError("Training, validation, and test sets must be non-empty.")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
validation_loader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(validation_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Class order: {train_dataset.classes}")

# =====================================================
# MODEL
# =====================================================
model = models.mobilenet_v2(weights="DEFAULT")
model.classifier[1] = nn.Linear(
    model.last_channel, len(CLASS_NAMES)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

# =====================================================
# TRAIN AND SELECT BY VALIDATION ACCURACY
# =====================================================
best_validation_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * train_correct / max(train_total, 1)
    val_accuracy = 100 * val_correct / max(val_total, 1)
    average_loss = train_loss / max(train_total, 1)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Validation Acc: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_validation_accuracy:
        best_validation_accuracy = val_accuracy
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  Saved best experimental model: {BEST_MODEL_PATH}")

# =====================================================
# TEST BEST EXPERIMENTAL CHECKPOINT
# =====================================================
print("\nLoading best experimental checkpoint...")
model.load_state_dict(
    torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
)
model.eval()

all_predictions = []
all_labels = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        predictions = model(images).argmax(1).cpu().tolist()
        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

print("\nEXPERIMENTAL TEST RESULTS")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4
))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    all_labels, all_predictions,
    labels=list(range(len(CLASS_NAMES)))
))

print(f"\nExperimental checkpoint: {BEST_MODEL_PATH}")
print(f"Best validation accuracy: {best_validation_accuracy:.2f}%")
print("Original best_model.pth was not modified.")
