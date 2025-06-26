import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0 # Import the pre-trained model
from tensorflow.keras.optimizers import Adam # Import Adam optimizer

# <--- IMPORTANT: Import from config_transfer.py
from config_transfer import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, LEARNING_RATE

def build_transfer_learning_model():
    """
    Builds a CNN model using EfficientNetB0 for transfer learning (binary classification).
    """
    # 1. Load the pre-trained EfficientNetB0 base model
    #    - weights='imagenet': Use weights pre-trained on ImageNet.
    #    - include_top=False: Exclude the original classification head of EfficientNet,
    #                         we'll add our own.
    #    - input_shape: Must match the input expected by the model (224, 224, 3 for EfficientNetB0).
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) # (224, 224, 3) as per config_transfer.py
    )

    # 2. Freeze the layers of the base model
    #    This prevents the pre-trained weights from being updated during initial training,
    #    using them as fixed feature extractors.
    base_model.trainable = False

    # 3. Build the new classification head on top of the base model
    model = models.Sequential([
        base_model, # The pre-trained EfficientNetB0 model
        layers.GlobalAveragePooling2D(), # Reduces spatial dimensions to a single vector
        layers.Dense(256, activation='relu'), # A dense layer for feature combination
        layers.Dropout(0.5), # Regularization
        layers.Dense(NUM_CLASSES, activation='sigmoid') # Output layer for 2 classes (RA/Healthy)
    ])

    # 4. Compile the model
    #    Using a lower learning rate is common for transfer learning.
    #    clipnorm is added for numerical stability.
    optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0) # <--- Added clipnorm for stability
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy', # For 2 classes
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc_roc') # AUC-ROC score
        ]
    )

    return model

if __name__ == '__main__':
    # This block allows you to test the model definition independently.
    print("Building and summarizing the Transfer Learning CNN model (EfficientNetB0)...")
    model = build_transfer_learning_model() # Call the transfer learning model builder
    model.summary()
    print("\nModel summary complete. The transfer learning model is ready to be used for training.")