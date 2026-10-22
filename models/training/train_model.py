import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2

# =====================================================
# IMPORT PROJECT FILES
# =====================================================

CURRENT_FILE = Path(__file__).resolve()

sys.path.append(str(CURRENT_FILE.parents[1] / "utils"))
sys.path.append(str(CURRENT_FILE.parent))

from constants import (
    NUM_CLASSES,
    EPOCHS,
    LEARNING_RATE,
    SAVED_MODEL_PATH
)

from data_loader import (
    train_loader,
    validation_loader
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing Device : {device}")

# =====================================================
# MODEL
# =====================================================

model = mobilenet_v2(weights="DEFAULT")

model.classifier[1] = nn.Linear(
    model.last_channel,
    NUM_CLASSES
)

model = model.to(device)

# =====================================================
# LOSS & OPTIMIZER
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =====================================================
# CREATE SAVE DIRECTORY
# =====================================================

SAVED_MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)

best_validation_accuracy = 0.0

# =====================================================
# TRAINING LOOP
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = (correct / total) * 100

    # ==========================
    # VALIDATION
    # ==========================

    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)

            validation_correct += (predicted == labels).sum().item()

    validation_accuracy = (
        validation_correct /
        validation_total
    ) * 100

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        torch.save(
            model.state_dict(),
            SAVED_MODEL_PATH / "best_model.pth"
        )

print("\nTraining Completed Successfully!")

print(f"Best Validation Accuracy : {best_validation_accuracy:.2f}%")

print(f"Model Saved To : {SAVED_MODEL_PATH / 'best_model.pth'}")