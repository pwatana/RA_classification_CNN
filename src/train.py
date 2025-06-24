import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # Use tensorflow.keras for callbacks
import numpy as np # Import numpy for random seed setting

from config import MODELS_DIR, MODEL_NAME, EPOCHS, HISTORY_NAME, RANDOM_SEED, BATCH_SIZE
from data_loader import split_and_copy_data_from_csv, get_image_data_generators
from model import build_cnn_model

def train_model():
    """
    Trains the CNN model using the prepared data generators.
    """
    print("Starting model training...")

    # Set random seeds for reproducibility across runs
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED) # For hash-based operations

    # 1. Prepare Data: This will trigger the data splitting and copying
    #    to data/processed, and set up the ImageDataGenerators.
    print("\n--- Preparing Data ---")
    split_and_copy_data_from_csv() # This populates data/processed
    train_generator, validation_generator, test_generator = get_image_data_generators()
    print(f"Data Generators Ready: Train samples={train_generator.samples}, Val samples={validation_generator.samples}, Test samples={test_generator.samples}")


    # 2. Build Model
    print("\n--- Building Model ---")
    model = build_cnn_model()
    model.summary() # Print model summary before training


    # 3. Define Callbacks for Training
    print("\n--- Setting Up Callbacks ---")
    # Callback to save the best model weights based on validation accuracy
    model_checkpoint_path = os.path.join(MODELS_DIR, MODEL_NAME)
    checkpoint = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_accuracy', # Monitor validation accuracy
        save_best_only=True,    # Only save the best performing model
        mode='max',             # 'max' because we want to maximize accuracy
        verbose=1               # Log when a new best model is saved
    )

    # Callback for early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',     # Monitor validation loss
        patience=10,            # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True, # Restore model weights from the epoch with the best monitored value
        verbose=1
    )

    # Callback to reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',     # Monitor validation loss
        factor=0.2,             # Factor by which the learning rate will be reduced (new_lr = lr * factor)
        patience=5,             # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=0.00001,         # Lower bound on the learning rate
        verbose=1
    )

    callbacks = [checkpoint, early_stopping, reduce_lr]


    # 4. Train the Model
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        # steps_per_epoch is important if generator doesn't provide it automatically (ImageDataGenerator does)
        # It's calculated as total_samples // batch_size
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        # validation_steps is important for validation_data from generator
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    print("\nTraining complete.")

    # 5. Save Training History
    print("\n--- Saving Training History ---")
    history_df = pd.DataFrame(history.history)
    history_save_path = os.path.join(MODELS_DIR, HISTORY_NAME)
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved to {history_save_path}")

    # 6. Evaluate the Best Model on the Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    # Load the best model saved by ModelCheckpoint for final evaluation
    # This ensures we evaluate the model with the best validation performance
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