import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0 # <--- Import the pre-trained model
from tensorflow.keras.optimizers import Adam
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, LEARNING_RATE

def build_cnn_model(): # Renamed to build_transfer_learning_model or keep build_cnn_model and update its content
    """
    Builds a CNN model using EfficientNetB0 for transfer learning.
    """
    # 1.Load the pre-trained EfficientNetB0 base model
    base_model = EfficientNetB0(
        weights='imagenet',       # Use weights pre-trained on ImageNet
        include_top=False,        # Exclude the original classification head
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS) # This should be (224, 224, 3)
    )

    # 2. Freeze the layers of the base model
    #    This prevents the pre-trained weights from being updated during initial training.
    base_model.trainable = False

    # 3. Build the new classification head on top of the base model
    model = models.Sequential([
        base_model, # The pre-trained EfficientNetB0 model
        layers.GlobalAveragePooling2D(), # Reduces spatial dimensions to a single vector
        layers.Dense(256, activation='relu'), # A dense layer for feature combination
        layers.Dropout(0.5), # Regularization
        layers.Dense(NUM_CLASSES, activation='softmax') # Output layer for our 3 classes
    ])

    # 4. Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy', # For multi-class classification
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

if __name__ == '__main__':
    print("Building and summarizing the Transfer Learning CNN model (EfficientNetB0)...")
    model = build_cnn_model() # Call the updated function
    model.summary()
    print("\nModel summary complete. The transfer learning model is ready to be used for training.")