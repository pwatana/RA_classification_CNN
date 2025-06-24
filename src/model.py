import tensorflow as tf
from tensorflow.keras import layers, models # Use tensorflow.keras for this setup
from config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES, LEARNING_RATE

def build_cnn_model():
    """
    Builds a Convolutional Neural Network model for binary classification of X-ray images.
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

        # Output Layer
        # Use 'sigmoid' activation for binary classification (NUM_CLASSES=2)
        # The loss function will be 'binary_crossentropy' for this setup.
        layers.Dense(1, activation='sigmoid') # Correct for binary classification with binary_crossentropy    
    ])

    # Optimizer and Compilation
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy' if NUM_CLASSES == 2 else 'categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

if __name__ == '__main__':
    # This block allows you to test the model definition independently.
    print("Building and summarizing the CNN model defined in model.py...")
    model = build_cnn_model()
    model.summary()
    print("\nModel summary complete. The model is ready to be used for training.")