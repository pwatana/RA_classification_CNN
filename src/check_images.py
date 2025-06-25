import os
from PIL import Image, UnidentifiedImageError
import traceback # To print full stack trace if desired
# Import relevant paths from your project config
from config import PROCESSED_DATA_DIR, CLASS_NAMES # Assuming CLASS_NAMES is still used for subdirs if needed, else ['RA', 'Healthy']

def check_images_in_directory(directory_path):
    """
    Checks all image files in a given directory and its subdirectories for readability.
    Reports any errors encountered during opening/reading.
    """
    print(f"\n--- Checking images in: {directory_path} ---")
    problematic_files = []
    total_checked = 0

    # Walk through the directory (e.g., train/RA, train/Healthy)
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): # Check common image formats
                file_path = os.path.join(root, filename)
                total_checked += 1
                try:
                    with Image.open(file_path) as img:
                        img.verify() # Verify file integrity
                        # If you want to force a full read (might be slow but catches more I/O issues)
                        # img.load()
                    # print(f"  [OK] {file_path}") # Uncomment to see all successful files
                except (UnidentifiedImageError, IOError, TimeoutError, OSError) as e:
                    print(f"  [ERROR] Problematic file: {file_path} (Error: {e})")
                    problematic_files.append(file_path)
                except Exception as e:
                    print(f"  [UNEXPECTED ERROR] Problematic file: {file_path} (Error: {e})")
                    problematic_files.append(file_path)
    
    print(f"--- Finished checking {total_checked} images in {directory_path} ---")
    if problematic_files:
        print(f"Total problematic files found: {len(problematic_files)}")
    else:
        print("No problematic image files found.")
    return problematic_files


if __name__ == "__main__":
    print("Starting image integrity check across processed data directories.")
    all_problematic_images = []

    # Check train, val, and test directories
    data_subsets = ['train', 'val', 'test']
    
    for subset in data_subsets:
        subset_path = os.path.join(PROCESSED_DATA_DIR, subset)
        if not os.path.exists(subset_path):
            print(f"Directory {subset_path} does not exist. Please run data_loader.py first to prepare data.")
            continue
        
        # Check subdirectories within each subset (e.g., train/RA, train/Healthy)
        # Assuming CLASS_NAMES from config.py defines these subdirs
        for class_name in ['RA', 'Healthy']: # Using hardcoded class names for 2-class problem
            class_path = os.path.join(subset_path, class_name)
            if not os.path.exists(class_path):
                print(f"Class directory {class_path} does not exist. Skipping.")
                continue
            
            problem_files = check_images_in_directory(class_path)
            all_problematic_images.extend(problem_files)

    print("\n--- Image Check Complete ---")
    if all_problematic_images:
        print("\nSummary of all problematic images found:")
        for f in all_problematic_images:
            print(f"- {f}")
        print("\nPlease inspect these files. They might be corrupted or unreadable.")
    else:
        print("All images in processed data directories appear to be readable.")