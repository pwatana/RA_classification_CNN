import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight # Import for class weight calculation

from src.config import MODELS_DIR, MODEL_NAME, EPOCHS, HISTORY_NAME, RANDOM_SEED, BATCH_SIZE
from src.data_loader import split_and_copy_data_from_csv, get_image_data_generators # <--- Data loader should return counts now
from src.model import build_cnn_model

def train_model():
    """
    Trains the CNN model using the prepared data generators,
    incorporating class weights to handle imbalance.
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
        print(f"{data_type.capitalize()} set: RA={counts['RA']}, Healthy={counts['Healthy']}")

    print(f"\nData Generators Ready: Train samples={train_generator.samples}, Val samples={validation_generator.samples}, Test samples={test_generator.samples}")

    # 2. Calculate Class Weights
    # Get the true labels from the training generator to compute weights
    # ImageDataGenerator.classes provides the integer labels (0 or 1) for each image in order.
    # Note: train_generator.classes will give labels for the *entire* train set, which is what we need.
    labels_binary = train_generator.classes # These are 0s and 1s corresponding to Healthy and RA
    class_names = list(train_generator.class_indices.keys()) # e.g., ['Healthy', 'RA']
    class_indices = train_generator.class_indices # e.g., {'Healthy': 0, 'RA': 1}

    # Use sklearn.utils.class_weight.compute_class_weight
    # 'balanced' mode automatically computes weights inversely proportional to class frequencies.
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_binary),
        y=labels_binary
    )
    # Map the weights to a dictionary where keys are class indices (0, 1)
    class_weights = {class_indices[name]: weight for name, weight in zip(class_names, class_weights_array)}

    print("\n--- Calculated Class Weights ---")
    print(f"Class Weights: {class_weights}")
    # Example interpretation: If Healthy (0) has weight 0.5 and RA (1) has weight 1.8,
    # misclassifying RA will be penalized more.


    # 3. Build Model
    print("\n--- Building Model ---")
    model = build_cnn_model()
    model.summary()


    # 4. Define Callbacks for Training
    print("\n--- Setting Up Callbacks ---")
    model_checkpoint_path = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
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