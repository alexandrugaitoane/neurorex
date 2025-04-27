import tensorflow as tf
from tensorflow import keras

def ct_hemorrhage_classifier(input_shape=(224, 224, 3), num_classes=2):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    data_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomZoom(0.2)
    ])
    
    model = tf.keras.models.Sequential([
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model