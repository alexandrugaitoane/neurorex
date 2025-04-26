import tensorflow as tf
from tensorflow import keras


def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layer")

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d")(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_1")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_2")(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_3")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_4")(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_5")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2")(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv2d_6")(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv2d_7")(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3")(c4)

    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="conv2d_8")(p4)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation="relu", padding="same", name="conv2d_9")(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same", name="conv2d_transpose")(c5)
    u6 = tf.keras.layers.Concatenate(name="concatenate")([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv2d_10")(u6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="conv2d_11")(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same", name="conv2d_transpose_1")(c6)
    u7 = tf.keras.layers.Concatenate(name="concatenate_1")([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_12")(u7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="conv2d_13")(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same", name="conv2d_transpose_2")(c7)
    u8 = tf.keras.layers.Concatenate(name="concatenate_2")([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_14")(u8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="conv2d_15")(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same", name="conv2d_transpose_3")(c8)
    u9 = tf.keras.layers.Concatenate(name="concatenate_3")([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_16")(u9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2d_17")(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="conv2d_18")(c9)

    model = tf.keras.models.Model(inputs, outputs, name="functional")
    return model