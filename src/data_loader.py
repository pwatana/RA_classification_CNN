import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 # Import OpenCV for advanced image processing

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT, IMG_CHANNELS

def prepare_data_directories():
    """
    Creates and clears processed data directories for train, validation, and test sets.
    """
    for sub_dir in ['train', 'val', 'test']:
        for class_dir in ['RA', 'Healthy']: # Assuming these are your class names
            path = os.path.join(PROCESSED_DATA_DIR, sub_dir, class_dir)
            if os.path.exists(path):
                shutil.rmtree(path) # Clear previous data
            os.makedirs(path, exist_ok=True)

def split_and_copy_data_from_csv():
    """
    Reads image filenames and labels from data.csv, splits into train/val/test,
    and copies images to their respective processed directories.
    """
    print("Splitting and copying data based on data.csv...")
    prepare_data_directories()

    csv_path = os.path.join(RAW_DATA_DIR, 'data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"data.csv not found at {csv_path}. Please ensure it's in the data/raw directory.")

    df = pd.read_csv(csv_path)

    # Assuming your CSV has 'Image' for filename and 'Label' for classification
    # Adjust column names if they are different in your actual data.csv
    image_filenames = df['Image'].tolist()
    labels = df['Label'].tolist()

    unique_labels = sorted(list(set(labels)))
    if not (len(unique_labels) == 2 and 'RA' in unique_labels and 'Healthy' in unique_labels):
        print(f"Warning: Unexpected labels found in data.csv: {unique_labels}. Expected 'RA' and 'Healthy'.")

    # Create full paths for images
    full_image_paths = [os.path.join(RAW_DATA_DIR, fname) for fname in image_filenames]

    # First split: train+val vs test (stratified by label)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        full_image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    # Second split: train vs val from train_val set (stratified by label)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), # Adjust test_size for the second split
        random_state=RANDOM_SEED, stratify=y_train_val
    )

    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    for data_type, (images, labels_list) in datasets.items():
        print(f"Copying {data_type} images...")
        for i, img_path in enumerate(images):
            class_name = labels_list[i] # Directly use the label from CSV
            dest_dir = os.path.join(PROCESSED_DATA_DIR, data_type, class_name)
            os.makedirs(dest_dir, exist_ok=True) # Ensure class-specific dir exists
            try:
                shutil.copy(img_path, dest_dir)
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path}. Skipping.")
    print("Data splitting and copying complete.")

def custom_image_preprocessing(img_array):
    """
    Applies a series of preprocessing steps to an image array:
    1. Grayscale conversion (if input is multi-channel)
    2. Gaussian Denoising
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalization to [0, 1]

    Args:
        img_array (np.array): A NumPy array representing the image.
                             Expected shape (height, width, channels) or (height, width).
                             ImageDataGenerator provides images as float32 by default.

    Returns:
        np.array: The processed image array, normalized to [0, 1] and with correct channel dimension.
    """
    # Convert to uint8 for OpenCV operations (important!)
    # ImageDataGenerator provides images as float32 in range [0, 255] (or [0, 1] if rescale is set).
    # Since we don't use rescale=1./255 in ImageDataGenerator, assume input is 0-255 range.
    img_array_uint8 = img_array.astype(np.uint8)

    # 1. Grayscale Conversion (if not already grayscale)
    if len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 3: # Check if it's an RGB image
        img_gray = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2GRAY)
    elif len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 1:
        img_gray = np.squeeze(img_array_uint8, axis=-1) # Remove channel dim if already grayscale (H, W, 1) -> (H, W)
    else:
        img_gray = img_array_uint8 # Already grayscale or single channel (H, W)

    # 2. Gaussian Denoising
    # Kernel size (e.g., 5x5) and sigma (0 for auto)
    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ClipLimit: threshold for contrast limiting. Larger values give more contrast.
    # tileGridSize: size of grid for histogram equalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # 4. Normalization to [0, 1]
    # Since we set rescale=None in ImageDataGenerator, we normalize here.
    img_normalized = img_clahe / 255.0

    # Ensure the output has the correct channel dimension for Keras (e.g., (height, width, 1) for grayscale)
    # The model expects a specific input shape. If IMG_CHANNELS is 1, add a channel dimension.
    if IMG_CHANNELS == 1:
        img_normalized = np.expand_dims(img_normalized, axis=-1) # Add channel dimension (H, W) -> (H, W, 1)

    return img_normalized


def get_image_data_generators():
    """
    Creates and returns Keras ImageDataGenerators for training and validation/testing.
    Applies augmentation to the training data and custom preprocessing to all sets.
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        # rescale=None, # Normalization is now handled in custom_image_preprocessing
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=custom_image_preprocessing # Apply custom preprocessing here
    )

    # Validation and test data (only custom preprocessing, no augmentation)
    val_test_datagen = ImageDataGenerator(
        # rescale=None, # Normalization is now handled in custom_image_preprocessing
        preprocessing_function=custom_image_preprocessing # Apply custom preprocessing here
    )

    # Note: color_mode='rgb' is used here because flow_from_directory reads images as RGB by default.
    # Our custom_image_preprocessing function then handles the conversion to grayscale (1 channel).
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary', # For binary classification (RA or Healthy)
        seed=RANDOM_SEED
    )

    validation_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False, # Do not shuffle test data for consistent evaluation
        seed=RANDOM_SEED
    )

    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    print("Running data_loader.py as main script for testing...")
    try:
        # First, ensure you have placed your data.csv and images in the data/raw/ directory.
        split_and_copy_data_from_csv()
        train_gen, val_gen, test_gen = get_image_data_generators()
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}") # Should show {'Healthy': 0, 'RA': 1} or similar

        # Optional: Verify one batch of images and their shape
        print("\nVerifying image batch shape and type after preprocessing...")
        first_batch_images, first_batch_labels = next(train_gen)
        print(f"Shape of image batch: {first_batch_images.shape}")
        print(f"Data type of image batch: {first_batch_images.dtype}")
        print(f"Pixel min value: {first_batch_images.min()}, max value: {first_batch_images.max()}")
        if first_batch_images.shape[-1] != IMG_CHANNELS:
            print(f"Warning: Expected {IMG_CHANNELS} channel(s) but got {first_batch_images.shape[-1]}. Check IMG_CHANNELS in config.py.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'data.csv' and all image files are placed in the 'data/raw/' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")