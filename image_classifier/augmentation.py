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
    plot_level=plot_level,
    name="data_augmentation_layer",
    data=None,
    data_object="",
):
    """create augmentation layer.

    Args:
        factor (float, optional): augmentation factor. Defaults to 0.2.
        plot_level (int, optional): plot level. Defaults to plot_level.
        name (str, optional): layer name. Defaults to "data_augmentation_layer".
        data (tf.keras.preprocessing.image_dataset, optional): data. Defaults to None.
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
        target_class = random.choice(data.class_names)
        target_dir = os.path.join(data_object, "train", target_class)
        random_image = random.choice(os.listdir(target_dir))
        random_image_path = target_dir + "/" + random_image

        plt.figure(figsize=(10, 15))

        image = mpimg.imread(random_image_path)
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"The original random image from class: {target_class}")
        plt.axis(False)

        augmented_image = layer(tf.expand_dims(image, axis=0), training=True)
        plt.subplot(1, 2, 2)
        plt.imshow(tf.squeeze(augmented_image / 255.0))
        plt.title(f"Augmented Image")
        plt.axis(False)

    return layer
