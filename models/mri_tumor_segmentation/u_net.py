import tensorflow as tf
from tensorflow import keras

def unet_model(input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(input_shape)

    # Contracting path
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Expanding path
    u6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = tf.keras.models.Model(inputs, outputs)
    return model
