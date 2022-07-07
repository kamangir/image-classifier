import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
from abcli import *
from abcli import logging
import logging

logger = logging.getLogger(__name__)


def create_layer(
    factor=0.2,
    name="data_augmentation_layer",
    data_object="",
    log_level=log_level,
    plot_level=plot_level,
):
    """create augmentation layer.

    Args:
        factor (float, optional): augmentation factor. Defaults to 0.2.
        name (str, optional): layer name. Defaults to "data_augmentation_layer".
        data_object (str, optional): data object. Defaults to "".

    Returns:
        tf.keras.Sequential: augmentation layer.
    """
    layer = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor),
            tf.keras.layers.experimental.preprocessing.RandomZoom(factor),
            tf.keras.layers.experimental.preprocessing.RandomHeight(factor),
            tf.keras.layers.experimental.preprocessing.RandomWidth(factor),
            # tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
            # tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        ],
        name=name,
    )

    if plot_level >= PLOT_ON:
        data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_object, "train"),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        target_class = random.choice(data.class_names)
        target_dir = os.path.join(data_object, "train", target_class)

        image = mpimg.imread(
            os.path.join(target_dir, random.choice(os.listdir(target_dir)))
        )

        plt.figure(figsize=(10, 15))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"random {target_class}")
        plt.axis(False)

        plt.subplot(1, 2, 2)
        plt.imshow(
            tf.squeeze(layer(tf.expand_dims(image, axis=0), training=True) / 255.0)
        )
        plt.title("augmentation output")
        plt.axis(False)

    return layer
