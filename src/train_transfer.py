import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # <--- NEW: Import Adam optimizer here

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix

from config_transfer import MODELS_DIR, MODEL_NAME, EPOCHS, HISTORY_NAME, RANDOM_SEED, BATCH_SIZE, LEARNING_RATE # Import LEARNING_RATE
from data_loader_transfer import split_and_copy_data_from_csv, get_image_data_generators
from model_transfer import build_transfer_learning_model # Use the transfer learning model builder

# Define a separate, lower learning rate for fine-tuning
FINE_TUNE_LEARNING_RATE = LEARNING_RATE / 10 # Start with 1/10th of initial LR
FINE_TUNE_EPOCHS = 20 # Number of epochs for fine-tuning phase (can be adjusted)

def train_model(model_to_use=None):
    """
    Trains the Transfer Learning CNN model in two phases:
    1. Train only the head (base model frozen).
    2. Fine-tune the top layers of the base model (unfrozen).
    """
    print("Starting model training (Transfer Learning pipeline)...")

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    print("\n--- Preparing Data ---")
    class_distribution_counts = split_and_copy_data_from_csv()
    train_generator, validation_generator, test_generator = get_image_data_generators()

    print("\n--- Class Distribution in Processed Data ---")
    for data_type, counts in class_distribution_counts.items():
        counts_str = ", ".join([f"{name}={count}" for name, count in counts.items()])
        print(f"{data_type.capitalize()} set: {counts_str}")

    print(f"\nData Generators Ready: Train samples={train_generator.samples}, Val samples={validation_generator.samples}, Test samples={test_generator.samples}")

    print("\n--- Calculated Class Weights (for Balanced Dataset) ---")
    labels_integer = train_generator.classes
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_integer),
        y=labels_integer
    )
    class_weights = {train_generator.class_indices[name]: weight for name, weight in zip(train_generator.class_indices.keys(), class_weights_array)}
    print(f"Class Names and Indices: {train_generator.class_indices}")
    print(f"Calculated Class Weights: {class_weights}")
    # Note: Since oversampling makes it balanced, class_weights will be {0: 1.0, 1: 1.0}.
    # We will not apply explicit class_weight in model.fit for balanced data.

    model = model_to_use
    if model is None:
        print("\n--- Building New Model (Phase 1: Frozen Base) ---")
        model = build_transfer_learning_model() # This model is built with base_model.trainable=False
        
        # Initial compilation is already done inside build_transfer_learning_model
        # Print summary after initial build
        model.summary()

        # 4. Define Callbacks for Training (Phase 1)
        print("\n--- Setting Up Callbacks for Phase 1 ---")
        model_path_for_checkpoint_phase1 = os.path.join(MODELS_DIR, "phase1_" + MODEL_NAME) # Unique name for phase 1
        checkpoint_phase1 = ModelCheckpoint(
            model_path_for_checkpoint_phase1,
            monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        )
        early_stopping_phase1 = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1) # Shorter patience for initial phase
        reduce_lr_phase1 = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        callbacks_phase1 = [checkpoint_phase1, early_stopping_phase1, reduce_lr_phase1]

        # 5. Train the Model (Phase 1: Train Head Only)
        print("\n--- Starting Model Training (Phase 1: Train Head Only) ---")
        try:
            history_phase1 = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                epochs=EPOCHS, # Use main EPOCHS for initial training
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                callbacks=callbacks_phase1,
                # No explicit class_weight here as data is balanced by oversampling
                workers=1, use_multiprocessing=False, max_queue_size=5
            )
            print("\nPhase 1 Training complete.")
            # Load best weights from phase 1 for starting phase 2
            model.load_weights(model_path_for_checkpoint_phase1)
        except Exception as e:
            print(f"An error occurred during Phase 1 training: {e}")
            traceback.print_exc()
            print("Skipping Phase 2 due to Phase 1 error.")
            return # Exit if Phase 1 fails


    # --- Phase 2: Fine-Tuning the Entire Model ---
    print("\n--- Starting Phase 2: Fine-Tuning Base Model ---")

    # Unfreeze the base model (or a portion of it)
    model.layers[0].trainable = True # base_model is the first layer of sequential model

    # It's common to unfreeze only certain layers for very large models.
    # For EfficientNetB0, unfreezing the whole base can be okay, or just the top blocks.
    # For simplicity, unfreezing the whole base here.
    # Alternatively, you can iterate:
    # for layer in model.layers[0].layers[-20:]: # Unfreeze last 20 layers of base_model
    #    layer.trainable = True

    # Recompile the model with a much lower learning rate for fine-tuning
    # This is crucial for avoiding catastrophic forgetting of pre-trained features.
    fine_tune_optimizer = Adam(learning_rate=FINE_TUNE_LEARNING_RATE, clipnorm=1.0)
    model.compile(
        optimizer=fine_tune_optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc')
        ]
    )
    model.summary() # Show trainable params after unfreezing

    # Callbacks for Phase 2
    model_path_for_checkpoint_phase2 = os.path.join(MODELS_DIR, MODEL_NAME) # Save final best model here
    checkpoint_phase2 = ModelCheckpoint(
        model_path_for_checkpoint_phase2,
        monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    )
    early_stopping_phase2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr_phase2 = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001, verbose=1) # Even lower min_lr
    callbacks_phase2 = [checkpoint_phase2, early_stopping_phase2, reduce_lr_phase2]

    try:
        history_phase2 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=FINE_TUNE_EPOCHS, # Use specific epochs for fine-tuning
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks_phase2,
            # No explicit class_weight
            workers=1, use_multiprocessing=False, max_queue_size=5
        )
        print("\nPhase 2 Fine-Tuning complete.")
        # Combine histories if needed for plotting later
        # history = {**history_phase1.history, **history_phase2.history} # Simple merge, needs careful key handling
        # For simplicity, saving combined history from train.py's own history objects.
    except Exception as e:
        print(f"An error occurred during Phase 2 fine-tuning: {e}")
        traceback.print_exc()

    print("\nTraining complete.")

    # 6. Save Training History (Consolidated)
    # It's good to save both phases' histories or a combined one.
    # For simplicity here, the last history (phase2) will overwrite.
    # If phase 1 failed, this will save its history.
    print("\n--- Saving Training History ---")
    if 'history_phase2' in locals():
        history_df = pd.DataFrame(history_phase2.history)
    elif 'history_phase1' in locals():
        history_df = pd.DataFrame(history_phase1.history)
    else:
        print("No history to save.")
        history_df = pd.DataFrame() # Empty DataFrame

    if not history_df.empty:
        history_save_path = os.path.join(MODELS_DIR, HISTORY_NAME)
        history_df.to_csv(history_save_path, index=False)
        print(f"Training history saved to {history_save_path}")
    else:
        print("No training history to save.")


    # 7. Evaluate the Best Model on the Test Set
    print("\n--- Evaluating Best Model on Test Set ---")
    try:
        best_model = tf.keras.models.load_model(model_path_for_checkpoint_phase2) # Final saved model
        print(f"Loaded best model from: {model_path_for_checkpoint_phase2} for final evaluation.")
        
        loss, accuracy, precision, recall, auc_roc = best_model.evaluate(test_generator)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test AUC-ROC: {auc_roc:.4f}")

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
        traceback.print_exc()
        print("Final evaluation skipped.")

    print("\nModel training and evaluation pipeline finished.")


if __name__ == '__main__':
    # --- Start of new conditional execution logic ---
    # Model name from config_transfer.py
    model_path = os.path.join(MODELS_DIR, MODEL_NAME)

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