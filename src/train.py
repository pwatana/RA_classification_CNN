# src/model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, LEARNING_RATE

def build_cnn_model():
    """
    Builds a Convolutional Neural Network model for multi-class classification (3 classes).
    """
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Flatten the output for the Dense layers
        layers.Flatten(),

        # Dense Block
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization to prevent overfitting

        # Output Layer: CHANGED for 3-class classification
        layers.Dense(NUM_CLASSES, activation='softmax') # <--- Changed units to NUM_CLASSES (3) and activation to 'softmax'
    ])

    # Optimizer and Compilation: CHANGED loss for 3-class classification
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', # <--- Changed loss to 'categorical_crossentropy'
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]) # Metrics remain

    return model

if __name__ == '__main__':
    print("Building and summarizing the CNN model defined in model.py...")
    model = build_cnn_model()
    model.summary()
    print("\nModel summary complete. The model is ready to be used for training.")