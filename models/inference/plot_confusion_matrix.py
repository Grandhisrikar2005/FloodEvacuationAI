
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "outputs" / "evaluation" / "expanded_model_test_report.txt"
OUTPUT_PATH = ROOT / "outputs" / "evaluation" / "expanded_confusion_matrix.png"

# Values copied from your completed evaluation
cm = np.array([
    [19, 1, 1, 0, 1, 1],
    [0, 11, 1, 2, 0, 0],
    [2, 0, 16, 3, 0, 1],
    [0, 2, 1, 14, 0, 0],
    [1, 0, 2, 0, 18, 0],
    [2, 1, 0, 1, 0, 11]
])

classes = [
    "Agriculture", "Barren", "Roads",
    "Urban", "Vegetation", "Water"
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)
display.plot(cmap="Blues", values_format="d", xticks_rotation=35)
plt.title("Expanded Model — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
plt.close()

print(f"Confusion matrix saved to: {OUTPUT_PATH}")
