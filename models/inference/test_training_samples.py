
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# =====================================================
# PATHS AND CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "dataset" / "split" / "train"
MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "best_model.pth"

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =====================================================
# LOAD MODEL
# =====================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found:\n{MODEL_PATH}"
        )

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(
        model.last_channel,
        len(CLASS_NAMES),
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# =====================================================
# SELECT AND TEST ONE IMAGE PER CLASS
# =====================================================

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{TRAIN_PATH}"
        )

    model = load_model()

    print("=" * 70)
    print("MOBILENETV2 — ORIGINAL TRAINING SAMPLE DIAGNOSTIC")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    correct = 0
    tested = 0

    with torch.inference_mode():
        for expected_class in CLASS_NAMES:
            class_dir = TRAIN_PATH / expected_class

            if not class_dir.exists():
                print(
                    f"\n{expected_class}: folder not found — "
                    f"{class_dir}"
                )
                continue

            images = sorted(
                path for path in class_dir.iterdir()
                if path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
                }
            )

            if not images:
                print(f"\n{expected_class}: no images found")
                continue

            image_path = images[0]

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    tensor = PREPROCESS(image).unsqueeze(0)

                tensor = tensor.to(DEVICE)

                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

                confidence, predicted_index = probabilities.max(dim=0)
                predicted_class = CLASS_NAMES[predicted_index.item()]
                confidence_pct = confidence.item() * 100

                is_correct = predicted_class == expected_class
                tested += 1
                correct += int(is_correct)

                status = "MATCH" if is_correct else "MISMATCH"

                print(f"\nExpected class : {expected_class}")
                print(f"Image          : {image_path.name}")
                print(f"Predicted class: {predicted_class}")
                print(f"Confidence     : {confidence_pct:.2f}%")
                print(f"Result         : {status}")

            except Exception as error:
                print(f"\nError testing {image_path.name}: {error}")

    print("\n" + "=" * 70)
    print(f"Samples tested : {tested}")
    print(f"Correct labels : {correct}")

    if tested:
        print(f"Sample match rate: {correct / tested * 100:.2f}%")

    print("=" * 70)
    print("Diagnostic complete.")
    print("This small sample is not a formal accuracy evaluation.")


if __name__ == "__main__":
    main()
