import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# =====================================================
# PROJECT PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Allow Python to find flood_risk package
sys.path.insert(0, str(BASE_DIR))

from flood_risk.risk_predictor import calculate_flood_risk


# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "saved_models"
    / "best_model.pth"
)


# =====================================================
# CLASSES
# =====================================================

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water"
]

NUM_CLASSES = len(CLASS_NAMES)


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device :", device)


# =====================================================
# LOAD MOBILE NET V2
# =====================================================

model = models.mobilenet_v2(weights=None)

model.classifier[1] = nn.Linear(
    model.last_channel,
    NUM_CLASSES
)


# =====================================================
# LOAD TRAINED WEIGHTS
# =====================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

print("Model Loaded Successfully")


# =====================================================
# IMAGE TRANSFORMATION
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# IMAGE PREDICTION
# =====================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted_class.item()

    confidence = confidence.item() * 100

    land_cover = CLASS_NAMES[predicted_class]

    return land_cover, confidence


# =====================================================
# FILE PICKER
# =====================================================

def select_image():

    root = tk.Tk()

    root.withdraw()

    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Select Satellite Image",

        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png"),
            ("JPG Files", "*.jpg"),
            ("PNG Files", "*.png"),
            ("All Files", "*.*")
        ]
    )

    root.destroy()

    return file_path


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print(" SATELLITE FLOOD RISK ANALYSIS")
    print("========================================")

    print()
    print("Opening image selector...")

    image_path = select_image()


    # -------------------------------------------------
    # Check image selection
    # -------------------------------------------------

    if not image_path:

        print()
        print("No image selected.")

        sys.exit()


    # -------------------------------------------------
    # Predict land cover
    # -------------------------------------------------

    land_cover, confidence = predict_image(
        Path(image_path)
    )


    # -------------------------------------------------
    # Display classification
    # -------------------------------------------------

    print()
    print("========================================")
    print("LAND-COVER CLASSIFICATION")
    print("========================================")

    print("Image      :", image_path)
    print("Class      :", land_cover)
    print(f"Confidence : {confidence:.2f}%")


    # -------------------------------------------------
    # Flood-related inputs
    # -------------------------------------------------

    print()
    print("========================================")
    print("FLOOD FACTORS")
    print("========================================")

    rainfall = float(
        input("Enter rainfall (mm): ")
    )

    water_distance = float(
        input("Enter distance from water (m): ")
    )

    elevation = float(
        input("Enter elevation (m): ")
    )

    drainage = float(
        input("Enter drainage score (0-10): ")
    )


    # -------------------------------------------------
    # Calculate flood risk
    # -------------------------------------------------

    risk_score, risk_level = calculate_flood_risk(

        land_cover=land_cover,

        rainfall_mm=rainfall,

        water_distance_m=water_distance,

        elevation_m=elevation,

        drainage_score=drainage
    )


    # -------------------------------------------------
    # Final result
    # -------------------------------------------------

    print()
    print("========================================")
    print("       FLOOD RISK RESULT")
    print("========================================")

    print("Land Cover       :", land_cover)
    print(f"Classification   : {confidence:.2f}%")

    print()
    print(f"Rainfall         : {rainfall} mm")
    print(f"Water Distance   : {water_distance} m")
    print(f"Elevation        : {elevation} m")
    print(f"Drainage Score   : {drainage}/10")

    print()
    print(f"Flood Risk Score : {risk_score:.2f}")
    print("Flood Risk Level :", risk_level)

    print("========================================")