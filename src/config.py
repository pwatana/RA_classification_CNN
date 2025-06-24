# src/config.py
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Model directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 1 # Keep 1 for now; will change to 3 for transfer learning later
NUM_CLASSES = 3 # <--- CHANGED: Now 3 classes (Healthy, Mild RA, Severe RA)

# RA Classification Score Bins (for 3 classes)
# scores < 50 => Healthy
# scores 50-100 => Mild RA
# scores > 100 => Severe RA
RA_SCORE_BINS = {
    'Healthy': {'max_score': 49.99},
    'Mild RA': {'min_score': 50.0, 'max_score': 100.0},
    'Severe RA': {'min_score': 100.01}
}
# Define class names in desired order for consistency (e.g., alphabetical or clinical progression)
# ImageDataGenerator assigns indices alphabetically by default if not specified.
CLASS_NAMES = ['Healthy', 'Mild RA', 'Severe RA'] # <--- IMPORTANT: Defines class order

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1 # This will be taken from the remaining after validation split
RANDOM_SEED = 42

# Model parameters
MODEL_NAME = 'ra_classifier_cnn.h5'
HISTORY_NAME = 'training_history.csv'