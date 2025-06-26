import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix # Import for F1 and Specificity calculations
import traceback # For printing full tracebacks

# <--- IMPORTANT: Import from config_transfer.py
from src.config_transfer import MODELS_DIR, MODEL_NAME, EPOCHS, HISTORY_NAME, RANDOM_SEED, BATCH_SIZE
# <--- IMPORTANT: Import from data_loader_transfer.py and model_transfer.py
from src.data_loader_transfer import split_and_copy_data_from_csv, get_image_data_generators
from src.model_transfer import build_transfer_learning_model # Use the transfer learning model builder


def train_model(model_to_use=None):
    """
    Trains the Transfer Learning CNN model using prepared data generators and class weights.
    Can accept a pre-built or pre-loaded model.
    """
    print("Starting model training (Transfer Learning pipeline)...")

    # Set random seeds for reproducibility across runs
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    # 1. Prepare Data: This will trigger data splitting, copying, and return class counts.
    #    It will be fast if data/processed_transfer already exists and is populated.
    print("\n--- Preparing Data ---")
    class_distribution_counts = split_and_copy_data_from_csv()
    train_generator, validation_generator, test_generator = get_image_data_generators()

    print("\n--- Class Distribution in Processed Data ---")
    for data_type, counts in class_distribution_counts.items():
        counts_str = ", ".join([f"{name}={count}" for name, count in counts.items()])
        print(f"{data_type.capitalize()} set: {counts_str}")

    print(f"\nData Generators Ready: Train samples={train_generator.samples}, Val samples={validation_generator.samples}, Test samples={test_generator.samples}")

    # 2. Calculate Class Weights for 2 classes
    # With effective oversampling, this should yield {0: 1.0, 1: 1.0}
    print("\n--- Calculated Class Weights ---")
    labels_integer = train_generator.classes # This is a 1D array of integers (0 for Healthy, 1 for RA)

    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_integer),
        y=labels_integer
    )
    class_weights = {train_generator.class_indices[name]: weight for name, weight in zip(train_generator.class_indices.keys(), class_weights_array)}

    print(f"Class Names and Indices: {train_generator.class_indices}")
    print(f"Calculated Class Weights: {class_weights}")


    # 3. Use provided model or build a new one
    model = model_to_use
    if model is None:
        print("\n--- Building Model ---")
        model = build_transfer_learning_model() # <--- Build the transfer learning model
    
    if model: # Ensure model was successfully created/loaded
        model.summary()

        # 4. Define Callbacks for Training
        print("\n--- Setting Up Callbacks ---")
        model_path_for_checkpoint = os.path.join(MODELS_DIR, MODEL_NAME)
        checkpoint = ModelCheckpoint(
            model_path_for_checkpoint,
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


        # 5. Train the Model
        print("\n--- Starting Model Training ---")
        try:
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                epochs=EPOCHS,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                callbacks=callbacks,
                # class_weight=class_weights, # <--- Can keep or comment out if weights are 1.0/1.0
                workers=1,
                use_multiprocessing=False,
                max_queue_size=5
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
                best_model = tf.keras.models.load_model(model_path_for_checkpoint)
                print(f"Loaded best model from: {model_path_for_checkpoint} for final evaluation.")
                
                loss, accuracy, precision, recall, auc_roc = best_model.evaluate(test_generator)
                print(f"Test Loss: {loss:.4f}")
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Test Precision: {precision:.4f}")
                print(f"Test Recall: {recall:.4f}")
                print(f"Test AUC-ROC: {auc_roc:.4f}")

                # --- Calculate additional metrics (F1-Score, Specificity) ---
                print("\n--- Calculating Advanced Test Metrics ---")
                test_generator.reset()
                y_true = test_generator.classes
                y_pred_probs = best_model.predict(test_generator)
                y_pred_binary = (y_pred_probs > 0.5).astype(int)

                f1 = f1_score(y_true, y_pred_binary, pos_label=1)
                print(f"Test F1-Score (for RA class): {f1:.4f}")

                cm = confusion_matrix(y_true, y_pred_binary)
                TN = cm[0, 0]
                FP = cm[0, 1]
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                print(f"Test Specificity (for Healthy class): {specificity:.4f}")


            except Exception as e:
                print(f"An error occurred during evaluation or loading best model: {e}")
                print("Evaluating the last trained model instead (if checkpoint failed).")
                loss, accuracy, precision, recall, auc_roc = model.evaluate(test_generator)
                print(f"Test Loss (last epoch model): {loss:.4f}")
                print(f"Test Accuracy (last epoch model): {accuracy:.4f}")
                print(f"Test Precision (last epoch model): {precision:.4f}")
                print(f"Test Recall (last epoch model): {recall:.4f}")
                print(f"Test AUC-ROC (last epoch model): {auc_roc:.4f}")
                
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            traceback.print_exc()
    else:
        print("Model could not be built or loaded. Exiting training process.")

    print("\nModel training and evaluation pipeline finished.")


if __name__ == '__main__':
    # --- Start of new conditional execution logic ---
    model_path = os.path.join(MODELS_DIR, MODEL_NAME) # Using MODEL_NAME from config_transfer

    loaded_model = None
    if os.path.exists(model_path):
        load_option = input(f"\nModel '{MODEL_NAME}' found at '{MODELS_DIR}'. Load (L) to resume/fine-tune or Build new (B)? (L/B): ").strip().upper()
        if load_option == 'L':
            try:
                loaded_model = tf.keras.models.load_model(model_path)
                print(f"Loaded existing model from {model_path}.")
            except Exception as e:
                print(f"Error loading model: {e}. Proceeding to build a new model.")
        else:
            print("User chose to build a new model.")
    else:
        print(f"No existing model '{MODEL_NAME}' found at '{MODELS_DIR}'. A new model will be built.")

    # Call train_model with the potentially loaded model (or None if a new one is to be built)
    train_model(model_to_use=loaded_model)
    # --- End of new conditional execution logic ---