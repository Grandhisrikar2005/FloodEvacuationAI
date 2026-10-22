import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

from PIL import Image

# =====================================================
# IMPORT PROJECT FILES
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

sys.path.append(str(CURRENT_FILE.parents[1] / "utils"))

from constants import (
    NUM_CLASSES,
    CLASS_NAMES,
    SAVED_MODEL_PATH
)

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
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# IMAGE PATH
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parents[2]

IMAGE_PATH = PROJECT_ROOT / "dataset" / "sample_tile.jpg"

# =====================================================
# LOAD IMAGE
# =====================================================

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0).to(device)

# =====================================================
# PREDICTION
# =====================================================

with torch.no_grad():

    output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

predicted_class = CLASS_NAMES[predicted.item()]

print("\n==============================")
print("IMAGE PREDICTION")
print("==============================")

print(f"Image      : {IMAGE_PATH.name}")
print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence.item()*100:.2f}%")