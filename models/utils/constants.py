from pathlib import Path

# =====================================================
# PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "split"

TRAIN_PATH = DATASET_PATH / "train"
VALIDATION_PATH = DATASET_PATH / "validation"
TEST_PATH = DATASET_PATH / "test"

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "checkpoints"
SAVED_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models"

# =====================================================
# MODEL CONFIGURATION
# =====================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 0

NUM_CLASSES = 6

EPOCHS = 20

LEARNING_RATE = 0.001

# =====================================================
# CLASS NAMES
# =====================================================

CLASS_NAMES = [
    "Agriculture",
    "Barren",
    "Roads",
    "Urban",
    "Vegetation",
    "Water"
]