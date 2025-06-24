import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 # Import OpenCV for advanced image processing

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT, IMG_CHANNELS, RA_SCORE_BINS, CLASS_NAMES, NUM_CLASSES # <--- Import new config variables

def prepare_data_directories():
    """
    Creates and clears processed data directories for train, validation, and test sets.
    Now handles 3 class subdirectories.
    """
    for sub_dir in ['train', 'val', 'test']:
        for class_dir_name in CLASS_NAMES: # Use CLASS_NAMES for subdirectories
            path = os.path.join(PROCESSED_DATA_DIR, sub_dir, class_dir_name)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

def split_and_copy_data_from_csv():
    """
    Reads exam_number and score_avg from data.csv, derives image filenames and
    3-class labels (Healthy, Mild RA, Severe RA), splits, and copies images.
    """
    print("Splitting and copying data based on data.csv for 3 classes...")
    prepare_data_directories()

    csv_path = os.path.join(RAW_DATA_DIR, 'data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"data.csv not found at {csv_path}. Please ensure it's in the data/raw directory.")

    df = pd.read_csv(csv_path)

    if 'exam_number' not in df.columns:
        raise ValueError("Column 'exam_number' not found in data.csv. Please check your CSV header.")
    if 'score_avg' not in df.columns:
        raise ValueError("Column 'score_avg' not found in data.csv. Please check your CSV header.")

    df['image_filename'] = df['exam_number'].apply(lambda x: f"DX{x}.jpg")

    # 2. Derive 3-Class Labels (Healthy, Mild RA, Severe RA) based on RA_SCORE_BINS
    def assign_ra_class(score):
        if score < RA_SCORE_BINS['Healthy']['max_score']:
            return 'Healthy'
        elif RA_SCORE_BINS['Mild RA']['min_score'] <= score <= RA_SCORE_BINS['Mild RA']['max_score']:
            return 'Mild RA'
        elif score > RA_SCORE_BINS['Severe RA']['min_score']:
            return 'Severe RA'
        else:
            return 'Unknown' # Should not happen if bins cover all possibilities

    df['classification_label'] = df['score_avg'].apply(assign_ra_class)

    # Filter out any 'Unknown' labels if necessary, or ensure all data fits bins
    df = df[df['classification_label'] != 'Unknown']

    image_filenames = df['image_filename'].tolist()
    labels = df['classification_label'].tolist()

    all_image_paths = [os.path.join(RAW_DATA_DIR, fname) for fname in image_filenames]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT), # Make sure TEST_SPLIT is defined correctly
        random_state=RANDOM_SEED, stratify=y_train_val
    )

    datasets = {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    class_counts = {}

    for data_type, (images, labels_list) in datasets.items():
        print(f"Copying {data_type} images...")
        # Initialize counts for all classes in CLASS_NAMES
        current_type_counts = {name: 0 for name in CLASS_NAMES}
        for i, img_path in enumerate(images):
            class_name = labels_list[i]
            dest_dir = os.path.join(PROCESSED_DATA_DIR, data_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy(img_path, dest_dir)
                current_type_counts[class_name] += 1
            except FileNotFoundError:
                print(f"Warning: Image file not found at {img_path}. Skipping.")
        class_counts[data_type] = current_type_counts
        # Print counts for all 3 classes
        counts_str = ", ".join([f"{name}={count}" for name, count in current_type_counts.items()])
        print(f"  {data_type.capitalize()} set counts: {counts_str}")


    print("Data splitting and copying complete.")
    return class_counts


def custom_image_preprocessing(img_array):
    """
    Applies a series of preprocessing steps to an image array:
    1. Grayscale conversion (if input is multi-channel, though color_mode='grayscale' helps)
    2. Gaussian Denoising
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalization to [0, 1]
    """
    img_array_uint8 = img_array.astype(np.uint8)

    if len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 1:
        img_gray = np.squeeze(img_array_uint8, axis=-1)
    elif len(img_array_uint8.shape) == 2:
        img_gray = img_array_uint8
    else:
        img_gray = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2GRAY)

    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)
    img_normalized = img_clahe / 255.0
    # Ensure the output has the correct channel dimension for Keras (H, W, C)
    if IMG_CHANNELS == 1:
        img_normalized = np.expand_dims(img_normalized, axis=-1) # (H,W) -> (H,W,1)
    elif IMG_CHANNELS == 3:
        # Duplicate the grayscale channel 3 times to make it RGB-like for ImageNet models
        img_normalized = np.stack([img_normalized, img_normalized, img_normalized], axis=-1) # (H,W) -> (H,W,3)
    # The image generator's color_mode='grayscale' loads as (H,W,1), then squeezed to (H,W) for OpenCV.
    # So img_normalized here is (H,W). The stack operation ensures it becomes (H,W,3).

    return img_normalized # This line remains


def get_image_data_generators():
    """
    Creates and returns Keras ImageDataGenerators for training and validation/testing.
    Now configured for 3 classes.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest',
        preprocessing_function=custom_image_preprocessing
    )
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=custom_image_preprocessing
    )

    # Note the change to class_mode='categorical' and target_names=CLASS_NAMES
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical', # <--- CHANGED: categorical
        classes=CLASS_NAMES, # <--- IMPORTANT: Specify class names order
        seed=RANDOM_SEED
    )

    validation_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical', # <--- CHANGED: categorical
        classes=CLASS_NAMES, # <--- IMPORTANT: Specify class names order
        seed=RANDOM_SEED
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale',
        batch_size=BATCH_SIZE, class_mode='categorical', # <--- CHANGED: categorical
        classes=CLASS_NAMES, # <--- IMPORTANT: Specify class names order
        shuffle=False, seed=RANDOM_SEED
    )

    return train_generator, validation_generator, test_generator

if __name__ == '__main__':
    print("Running data_loader.py as main script for testing...")
    try:
        class_distribution = split_and_copy_data_from_csv()
        print("\n--- Class Distribution in Processed Data ---")
        for data_type, counts in class_distribution.items():
            counts_str = ", ".join([f"{name}={count}" for name, count in counts.items()])
            print(f"{data_type.capitalize()} set: {counts_str}")

        train_gen, val_gen, test_gen = get_image_data_generators()
        print(f"\nData Generators Ready: Train samples={train_gen.samples}, Val samples={val_gen.samples}, Test samples={test_gen.samples}")
        print(f"Class indices: {train_gen.class_indices}") # Should show {'Healthy': 0, 'Mild RA': 1, 'Severe RA': 2}

        print("\nVerifying image batch shape and type after preprocessing...")
        first_batch_images, first_batch_labels = next(train_gen)
        print(f"Shape of image batch: {first_batch_images.shape}")
        print(f"Data type of image batch: {first_batch_labels.dtype}") # Check label dtype
        print(f"Pixel min value: {first_batch_images.min()}, max value: {first_batch_images.max()}")
        if first_batch_images.shape[-1] != IMG_CHANNELS:
            print(f"Warning: Expected {IMG_CHANNELS} channel(s) but got {first_batch_images.shape[-1]}. Check IMG_CHANNELS in config.py.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'data.csv' and all image files are placed in the 'data/raw/' directory.")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please ensure 'exam_number' and 'score_avg' columns exist in your data.csv and are spelled correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")