import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Use tensorflow.keras for callbacks
import numpy as np
from sklearn.utils.class_weight import compute_class_weight # Import for class weight calculation

from config import MODELS_DIR, MODEL_NAME, EPOCHS, HISTORY_NAME, RANDOM_SEED, BATCH_SIZE, CLASS_NAMES # <--- Import CLASS_NAMES
from data_loader import split_and_copy_data_from_csv, get_image_data_generators
from model import build_cnn_model

def train_model():
    """
    Trains the CNN model using the prepared data generators,
    incorporating class weights to handle imbalance in multi-class setting.
    """
    print("Starting model training...")

    # Set random seeds for reproducibility across runs
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 1. Prepare Data: This will trigger data splitting, copying, and return class counts
    print("\n--- Preparing Data ---")
    class_distribution_counts = split_and_copy_data_from_csv() # Capture the returned counts
    train_generator, validation_generator, test_generator = get_image_data_generators()

    print("\n--- Class Distribution in Processed Data ---")
    for data_type, counts in class_distribution_counts.items():
        counts_str = ", ".join([f"{name}={count}" for name, count in counts.items()])
        print(f"{data_type.capitalize()} set: {counts_str}")

    print(f"\nData Generators Ready: Train samples={train_generator.samples}, Val samples={validation_generator.samples}, Test samples={test_generator.samples}")

    # 2. Calculate Class Weights for 3 classes
    print("\n--- Calculated Class Weights ---")
    # ImageDataGenerator.classes provides the integer labels (0, 1, 2) for each image in order
    labels_integer = train_generator.classes # This is a 1D array of integers corresponding to CLASS_NAMES order

    # Compute weights inversely proportional to class frequencies
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_integer), # Get unique class indices (0, 1, 2)
        y=labels_integer # Use the integer labels for computation
    )
    # Map the weights to a dictionary where keys are class indices (0, 1, 2)
    # The order of CLASS_NAMES in config.py is crucial for correct mapping.
    class_weights = {train_generator.class_indices[name]: weight for name, weight in zip(CLASS_NAMES, class_weights_array)}


    print(f"Class Names and Indices: {train_generator.class_indices}")
    print(f"Calculated Class Weights: {class_weights}")


    # 3. Build Model
    print("\n--- Building Model ---")
    model = build_cnn_model()
    model.summary()


    # 4. Define Callbacks for Training
    print("\n--- Setting Up Callbacks ---")
    model_checkpoint_path = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy', # Monitor validation accuracy
        save_best_only=True,    # Only save the best performing model
        mode='max',             # 'max' because we want to maximize accuracy
        verbose=1               # Log when a new best model is saved
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',     # Monitor validation loss
        patience=10,            # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Restore model weights from the epoch with the best monitored value
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',     # Monitor validation loss
        factor=0.2,             # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=5,             # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001,         # Lower bound on the learning rate
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr]


    # 5. Train the Model, now with class_weights
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks,
        class_weight=class_weights # <--- APPLY CLASS WEIGHTS HERE
    )

    print("\nTraining complete.")

    # 6. Save Training History
    print("\n--- Saving Training History ---")
    history_df = pd.DataFrame(history.history)
    history_save_path = os.path.join(MODELS_DIR, HISTORY_NAME)
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to {history_save_path}")

    # 7. Evaluate the Best Model on the Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    try:
        # Use tf.keras.models.load_model which should work for TF 2.x
        best_model = tf.keras.models.load_model(model_checkpoint_path)
        print(f"Loaded best model from: {model_checkpoint_path}")
        loss, accuracy, precision, recall = best_model.evaluate(test_generator)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
    except Exception as e:
        print(f"Could not load best model for evaluation: {e}")
        print("Evaluating the last trained model instead (if checkpoint failed).")
        loss, accuracy, precision, recall = model.evaluate(test_generator)
        print(f"Test Loss (last epoch model): {loss:.4f}")
        print(f"Test Accuracy (last epoch model): {accuracy:.4f}")
        print(f"Test Precision (last epoch model): {precision:.4f}")
        print(f"Test Recall (last epoch model): {recall:.4f}")

    print("\nModel training and evaluation pipeline finished.")


if __name__ == '__main__':
    train_model()