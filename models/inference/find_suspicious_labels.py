
from pathlib import Path
import csv
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset" / "expanded_v1"
CHECKPOINT = ROOT / "models" / "saved_models" / "expanded_best_model.pth"
OUTPUT = ROOT / "outputs" / "evaluation" / "suspicious_labels.csv"

CLASSES = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

rows = []

# Only inspect training and validation images.
# Keep the test set untouched for unbiased evaluation.
for split in ["train", "validation"]:
    split_dir = DATASET / split

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        true_label = class_dir.name

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    probabilities = torch.softmax(model(tensor), dim=1)[0]

                predicted_index = int(probabilities.argmax())
                predicted_label = CLASSES[predicted_index]
                confidence = float(probabilities[predicted_index])

                if predicted_label != true_label and confidence >= 0.70:
                    rows.append({
                        "split": split,
                        "filename": path.name,
                        "path": str(path.relative_to(ROOT)),
                        "folder_label": true_label,
                        "model_prediction": predicted_label,
                        "prediction_confidence": round(confidence, 4),
                        "review_status": "needs_review"
                    })

            except Exception as exc:
                print(f"Skipped {path}: {exc}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    fields = [
        "split", "filename", "path", "folder_label",
        "model_prediction", "prediction_confidence",
        "review_status"
    ]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSuspicious candidates: {len(rows)}")
print(f"Report saved to: {OUTPUT}")
print("No images, labels, or model weights were changed.")
print("These are candidates for review, not confirmed labeling errors.")
