import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed_transfer') # <--- NEW: Separate processed folder for TL
                                                                   # This avoids conflicts with simple CNN processed data
# Model directory
MODELS_DIR = os.path.join(BASE_DIR, 'models') # Keep models in the same folder

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3 # <--- CHANGED: 3 channels for transfer learning model (RGB-like input)
NUM_CLASSES = 2 # Still 2 classes (RA, Healthy) for binary

# RA Classification Threshold (For binary classification: score >= 50.0 is RA)
RA_SCORE_THRESHOLD = 50.0

# Define class names in desired order for consistency (ImageDataGenerator infers alphabetically by default)
CLASS_NAMES = ['Healthy', 'RA']

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50 # Adjust as needed
LEARNING_RATE = 0.001 # <--- Adjusted LEARNING_RATE for transfer learning (often lower)
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1 # This will be taken from the remaining after validation split
RANDOM_SEED = 42

# Model parameters
MODEL_NAME = 'ra_classifier_transfer_cnn.h5' # <--- NEW: Unique model name for TL
HISTORY_NAME = 'training_history_transfer.csv' # <--- NEW: Unique history name for TL
FILTERED_IMAGES_LOG = 'filtered_images_transfer.log' # <--- NEW: Unique log name for TL