
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = PROJECT_ROOT / "dataset" / "split" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "best_model.pth"

CLASS_NAMES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    dataset = datasets.ImageFolder(TEST_PATH, transform=transform)

    if dataset.classes != CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch!\n"
            f"Dataset classes: {dataset.classes}\n"
            f"Expected classes: {CLASS_NAMES}"
        )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    ))
    model.to(DEVICE)
    model.eval()

    actual, predicted = [], []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()
            predicted.extend(predictions)
            actual.extend(labels.tolist())

    print("=" * 65)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 65)
    print(f"Test images: {len(dataset)}")
    print(f"Device: {DEVICE}\n")

    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(
        actual, predicted, labels=list(range(len(CLASS_NAMES)))
    ))

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
