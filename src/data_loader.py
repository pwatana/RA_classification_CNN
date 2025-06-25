import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 # Import OpenCV for advanced image processing
from sklearn.utils import resample # Import for resampling
from PIL import Image, UnidentifiedImageError # Import for image validation


from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, RANDOM_SEED, VALIDATION_SPLIT, TEST_SPLIT, IMG_CHANNELS, RA_SCORE_THRESHOLD


def prepare_data_directories():
    """
    Creates and clears processed data directories for train, validation, and test sets.
    Now handles 2 class subdirectories: RA, Healthy.
    """
    for sub_dir in ['train', 'val', 'test']:
        for class_dir_name in ['RA', 'Healthy']: # Only 2 class subdirectories
            path = os.path.join(PROCESSED_DATA_DIR, sub_dir, class_dir_name)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)

def split_and_copy_data_from_csv():
    """
    Reads exam_number and score_avg from data.csv, derives image filenames and
    binary labels (RA/Healthy).
    Filters out unreadable image files.
    Performs oversampling on the training set, and copies images.
    """
    print("Splitting and copying data based on data.csv for 2 classes, with oversampling...")
    prepare_data_directories()

    csv_path = os.path.join(RAW_DATA_DIR, 'data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"data.csv not found at {csv_path}. Please ensure it's in the data/raw directory.")

    df = pd.read_csv(csv_path)

    if 'exam_number' not in df.columns:
        raise ValueError("Column 'exam_number' not found in data.csv. Please check your CSV header.")
    if 'score_avg' not in df.columns:
        raise ValueError("Column 'score_avg' not found in data.csv. Please check your CSV header.")

    # 1. Derive Image Filenames and Initial Labels
    df['image_filename'] = df['exam_number'].apply(lambda x: f"DX{x}.jpg")
    df['classification_label'] = df['score_avg'].apply(
        lambda score: 'RA' if score >= RA_SCORE_THRESHOLD else 'Healthy'
    )

    # --- NEW: Filter out unreadable or truly missing image files before splitting ---
    print("Pre-filtering for unreadable or missing image files...")
    all_image_paths_filtered = []
    all_labels_filtered = []
    skipped_by_filter_count = 0

    for index, row in df.iterrows():
        fname = row['image_filename']
        label = row['classification_label']
        full_path = os.path.join(RAW_DATA_DIR, fname)

        if not os.path.exists(full_path):
            # print(f"Pre-filter Warning: Image file not found at {full_path}. Skipping from dataset.") # Can uncomment for debug
            skipped_by_filter_count += 1
            continue # Skip to next row if file doesn't exist

        try:
            # Attempt to open the image with Pillow to check readability
            with Image.open(full_path) as img:
                img.verify() # Verify file integrity
            all_image_paths_filtered.append(full_path)
            all_labels_filtered.append(label)
        except (UnidentifiedImageError, IOError, SyntaxError) as e:
            # print(f"Pre-filter Warning: Cannot identify or read image file {full_path}. Skipping from dataset. Error: {e}") # Can uncomment for debug
            skipped_by_filter_count += 1
        except Exception as e:
            # print(f"Pre-filter Warning: Unexpected error reading {full_path}. Skipping. Error: {e}") # Can uncomment for debug
            skipped_by_filter_count += 1

    if skipped_by_filter_count > 0:
        print(f"Pre-filtering complete. {skipped_by_filter_count} unreadable/missing images skipped from CSV entries.")
    else:
        print("Pre-filtering complete. All images found and readable.")

    if not all_image_paths_filtered:
        raise ValueError("No readable image files found after pre-filtering. Cannot proceed.")
    # --- End NEW filtering section ---


    # Use the filtered_image_paths and filtered_labels for splitting
    # First split: train+val vs test (stratified by classification_label)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_image_paths_filtered, all_labels_filtered, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=all_labels_filtered
    )

    # Second split: train vs val from train_val set (stratified by classification_label)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        random_state=RANDOM_SEED, stratify=y_train_val
    )

    # --- Oversampling of the Training Set ---
    print("Performing oversampling on the training set...")
    train_df = pd.DataFrame({'image_path': X_train, 'label': y_train})

    df_majority = train_df[train_df.label == 'Healthy'] # Assuming 'Healthy' is the majority
    df_minority = train_df[train_df.label == 'RA']      # Assuming 'RA' is the minority

    majority_class_size = len(df_majority)

    df_minority_oversampled = resample(df_minority,
                                       replace=True,     # Sample with replacement (creates duplicates)
                                       n_samples=majority_class_size, # To match majority class size
                                       random_state=RANDOM_SEED) # For reproducibility

    df_oversampled_train = pd.concat([df_majority, df_minority_oversampled])

    # Shuffle the combined (oversampled) training dataset to mix the samples
    df_oversampled_train = df_oversampled_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    X_train_oversampled = df_oversampled_train['image_path'].tolist()
    y_train_oversampled = df_oversampled_train['label'].tolist()
    print(f"Oversampling complete. Training set size increased from {len(X_train)} to {len(X_train_oversampled)} samples.")
    print(f"  New Training set counts: RA={len(df_minority_oversampled)}, Healthy={len(df_majority)}")
    # --- End Oversampling Section ---


    # Now use the oversampled training data for the datasets dictionary
    datasets = {
        'train': (X_train_oversampled, y_train_oversampled), # Use oversampled data
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }

    class_counts = {}

    for data_type, (images, labels_list) in datasets.items():
        print(f"Copying {data_type} images...")
        current_type_counts = {'RA': 0, 'Healthy': 0}
        for i, img_path in enumerate(images):
            class_name = labels_list[i]
            dest_dir = os.path.join(PROCESSED_DATA_DIR, data_type, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            try:
                shutil.copy(img_path, dest_dir)
                current_type_counts[class_name] += 1
            except FileNotFoundError:
                # This should ideally not happen if pre-filtering works correctly,
                # but left as a safeguard.
                print(f"Warning: Image file not found at {img_path}. Skipping during copy phase.")
        class_counts[data_type] = current_type_counts
        print(f"  {data_type.capitalize()} set counts: RA={current_type_counts['RA']}, Healthy={current_type_counts['Healthy']}")


    print("Data splitting and copying complete.")
    return class_counts


def custom_image_preprocessing(img_array):
    """
    Applies a series of preprocessing steps to an image array:
    1. Grayscale conversion (from original RGB input if needed)
    2. Gaussian Denoising
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Normalization to [0, 1]
    5. Ensures 1 output channel.
    """
    img_array_uint8 = img_array.astype(np.uint8)

    # Convert original RGB input to Grayscale for OpenCV processing if color_mode='rgb' is used
    if len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 3:
        img_gray = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2GRAY)
    elif len(img_array_uint8.shape) == 3 and img_array_uint8.shape[2] == 1:
        img_gray = np.squeeze(img_array_uint8, axis=-1) # Already grayscale with channel dim, squeeze
    else: # Fallback or if already 2D
        img_gray = img_array_uint8


    # 2. Gaussian Denoising
    img_denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_denoised)

    # 4. Normalization to [0, 1]
    img_normalized_2d = img_clahe / 255.0 # This is (H,W)


    # 5. Ensure the output has 1 channel for Keras (H, W, 1)
    img_final = np.expand_dims(img_normalized_2d, axis=-1) # (H,W) -> (H,W,1)

    return img_final


def get_image_data_generators():
    """
    Creates and returns Keras ImageDataGenerators for training and validation/testing.
    Now configured for 2 classes and grayscale input.
    """
    train_datagen = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest',
        preprocessing_function=custom_image_preprocessing
    )

    val_test_datagen = ImageDataGenerator(
        preprocessing_function=custom_image_preprocessing
    )

    # REVERTED: color_mode='grayscale' and class_mode='binary'
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # REVERTED
        batch_size=BATCH_SIZE, class_mode='binary', # REVERTED
        seed=RANDOM_SEED
    )

    validation_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # REVERTED
        batch_size=BATCH_SIZE, class_mode='binary', # REVERTED
        seed=RANDOM_SEED
    )

    test_generator = val_test_datagen.flow_from_directory(
        directory=os.path.join(PROCESSED_DATA_DIR, 'test'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode='grayscale', # REVERTED
        batch_size=BATCH_SIZE, class_mode='binary', # REVERTED
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
        print(f"Class indices: {train_gen.class_indices}") # Will show {'Healthy': 0, 'RA': 1}

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