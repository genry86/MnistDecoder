import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import InputLayer

def build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        # tf.keras.Input(shape=input_shape),
        InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model