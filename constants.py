# FOLDER_PATH = r"C:\Users\zuzan\OneDrive\Pulpit\dataset\dataset\28x28 zbiorczo"
# IMAGES_SAVE_PATH = r"C:\Users\zuzan\OneDrive\Pulpit\dataset\dataset\28x28 images.npy"
# LABELS_SAVE_PATH = r"C:\Users\zuzan\OneDrive\Pulpit\dataset\dataset\28x28 labels.npy"

from pathlib import Path

DATA_PATH = Path('data')
DATA_PATH.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 26
