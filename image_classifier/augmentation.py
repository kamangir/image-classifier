import tensorflow as tf


def create_layer(factor=0.2):
    return tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor),
            tf.keras.layers.experimental.preprocessing.RandomZoom(factor),
            tf.keras.layers.experimental.preprocessing.RandomHeight(factor),
            tf.keras.layers.experimental.preprocessing.RandomWidth(factor),
            # tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
            # tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        ],
        name="data_augmentation_layer",
    )
