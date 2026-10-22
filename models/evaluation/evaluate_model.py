import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# =====================================================
# IMPORT PROJECT FILES
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

sys.path.append(str(CURRENT_FILE.parents[1] / "utils"))
sys.path.append(str(CURRENT_FILE.parents[1] / "training"))

from constants import (
    NUM_CLASSES,
    SAVED_MODEL_PATH
)

from data_loader import test_loader

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}")

# =====================================================
# LOAD MODEL
# =====================================================

model = mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        SAVED_MODEL_PATH / "best_model.pth",
        map_location=device
    )
)

model.to(device)

model.eval()

# =====================================================
# EVALUATION
# =====================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())

        all_labels.extend(labels.numpy())

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(all_labels, all_predictions)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print("\n===============================")
print("MODEL EVALUATION")
print("===============================")

print(f"Accuracy : {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall   : {recall*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    all_labels,
    all_predictions,
    digits=4
))