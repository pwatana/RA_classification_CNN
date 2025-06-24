import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Ensure this import is correct based on your Keras version
import cv2 # Import OpenCV for advanced image processing

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT, IMG_CHANNELS, RA_SCORE_THRESHOLD # <--- Import RA_SCORE_THRESHOLD

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
    Reads exam_number and score_avg from data.csv, derives image filenames and
    binary labels (RA/Healthy), splits into train/val/test,
    and copies images to their respective processed directories.
    """
    print("Splitting and copying data based on data.csv...")
    prepare_data_directories()

    csv_path = os.path.join(RAW_DATA_DIR, 'data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"data.csv not found at {csv_path}. Please ensure it's in the data/raw directory.")

    df = pd.read_csv(csv_path)

    # Validate required columns exist
    if 'exam_number' not in df.columns:
        raise ValueError("Column 'exam_number' not found in data.csv. Please check your CSV header.")
    if 'score_avg' not in df.columns:
        raise ValueError("Column 'score_avg' not found in data.csv. Please check your CSV header.")

    # 1. Derive Image Filenames: Assuming format DX<exam_number>.jpg
    # Example: exam_number 316153 -> DX316153.jpg
    df['image_filename'] = df['exam_number'].apply(lambda x: f"DX{x}.jpg")

    # 2. Derive Binary Labels (RA or Healthy) based on RA_SCORE_THRESHOLD
    # Adjust the threshold in config.py based on your clinical understanding.
    df['classification_label'] = df['score_avg'].apply(
        lambda score: 'RA' if score >= RA_SCORE_THRESHOLD else 'Healthy'
    )

    image_filenames = df['image_filename'].tolist()
    labels = df['classification_label'].tolist()

    # Create full paths for images
    all_image_paths = [os.path.join(RAW_DATA_DIR, fname) for fname in image_filenames]

    # First split: train+val vs test (stratified by classification_label)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    # Second split: train vs val from train_val set (stratified by classification_label)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
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
            class_name = labels_list[i] # Use the derived label
            dest_dir = os.path.join(PROCESSED_DATA_DIR, data_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy(img_path, dest_dir)
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path}. Skipping.")
    print("Data splitting and copying complete.")

def custom_image_preprocessing(img_array):
    """
    Applies a series of preprocessing steps to an image array:
    1. Grayscale conversion (if input is multi-channel, though color_mode='grayscale' helps)
    2. Gaussian Denoising
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalization to [0, 1]
    """
    # Convert to uint8 for OpenCV operations (important!)
    # img_array will now be (H, W, 1) if color_mode='grayscale' is used.
    img_array_uint8 = img_array.astype(np.uint8)

    # Grayscale Conversion: If input is (H, W, 1), squeeze to (H, W) for OpenCV 2D operations
    if len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 1:
        img_gray = np.squeeze(img_array_uint8, axis=-1) # Remove channel dim to get (H, W)
    elif len(img_array_uint8.shape) == 2:
        img_gray = img_array_uint8 # Already (H, W)
    else: # Fallback for unexpected shapes, e.g., if color_mode wasn't grayscale
        img_gray = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2GRAY) # Fallback to convert from RGB

    # 2. Gaussian Denoising
    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # 4. Normalization to [0, 1]
    img_normalized = img_clahe / 255.0

    # Ensure the output has the correct channel dimension for Keras (H, W, 1)
    # img_normalized is currently (H, W)
    img_normalized = np.expand_dims(img_normalized, axis=-1) # Add channel dimension

    return img_normalized


def get_image_data_generators():
    """
    Creates and returns Keras ImageDataGenerators for training and validation/testing.
    Applies augmentation to the training data and custom preprocessing to all sets.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=custom_image_preprocessing
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=custom_image_preprocessing
    )

    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # <--- CHANGED THIS TO 'grayscale'
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    validation_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # <--- CHANGED THIS TO 'grayscale'
        batch_size=BATCH_SIZE,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # <--- CHANGED THIS TO 'grayscale'
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=RANDOM_SEED
    )

    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    print("Running data_loader.py as main script for testing...")
    try:
        split_and_copy_data_from_csv()
        train_gen, val_gen, test_gen = get_image_data_generators()
        print(f"Train samples: {train_gen.samples}")
        print(f"Validation samples: {val_gen.samples}")
        print(f"Test samples: {test_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}")

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
    except ValueError as e: # Catch the new ValueError for missing columns
        print(f"Error: {e}")
        print("Please ensure 'exam_number' and 'score_avg' columns exist in your data.csv and are spelled correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")