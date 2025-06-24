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
IMG_CHANNELS = 1 # <--- REVERTED: 1 channel for grayscale
NUM_CLASSES = 2 # <--- REVERTED: 2 classes (RA, Healthy)

# RA Classification Threshold (For binary classification: score >= 50.0 is RA)
RA_SCORE_THRESHOLD = 50.0 # <--- REVERTED: Single threshold for binary split
# Removed RA_SCORE_BINS and CLASS_NAMES

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