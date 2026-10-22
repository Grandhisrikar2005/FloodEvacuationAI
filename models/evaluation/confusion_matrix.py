import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =====================================================
# IMPORT PROJECT FILES
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

sys.path.append(str(CURRENT_FILE.parents[1] / "utils"))
sys.path.append(str(CURRENT_FILE.parents[1] / "training"))

from constants import (
    NUM_CLASSES,
    SAVED_MODEL_PATH,
    CLASS_NAMES
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
# PREDICTIONS
# =====================================================

true_labels = []
predicted_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions.cpu().numpy())

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=True
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()