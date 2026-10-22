
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "expanded_v1" / "test"
CHECKPOINT = ROOT / "models" / "saved_models" / "expanded_best_model.pth"
OUTPUT = ROOT / "outputs" / "evaluation"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 16

CLASS_NAMES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

if not DATASET.exists():
    raise FileNotFoundError(f"Test dataset not found: {DATASET}")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

test_dataset = datasets.ImageFolder(DATASET, transform=transform)

if test_dataset.classes != sorted(CLASS_NAMES):
    raise ValueError(
        f"Class order mismatch. Found {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

state = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True,
)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

predictions = []
actual = []

with torch.inference_mode():
    for images, labels in test_loader:
        outputs = model(images.to(DEVICE))
        predictions.extend(outputs.argmax(1).cpu().tolist())
        actual.extend(labels.tolist())

accuracy = accuracy_score(actual, predictions)

report = classification_report(
    actual,
    predictions,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4,
)

matrix = confusion_matrix(
    actual,
    predictions,
    labels=list(range(len(CLASS_NAMES))),
)

print("\n===== EXPANDED MODEL TEST RESULTS =====")
print(f"Device: {DEVICE}")
print(f"Test images: {len(test_dataset)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(report)
print("Confusion matrix (rows=actual, columns=predicted):")
print(matrix)

report_path = OUTPUT / "expanded_model_test_report.txt"
with open(report_path, "w", encoding="utf-8") as file:
    file.write(f"Test images: {len(test_dataset)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    file.write(report)
    file.write("\nConfusion matrix (rows=actual, columns=predicted):\n")
    file.write(str(matrix))

print(f"\nReport saved to: {report_path}")
print("No training was performed. The checkpoint was not modified.")
print("Note: results measure agreement with existing test labels,")
print("which may themselves contain mislabeled satellite images.")
