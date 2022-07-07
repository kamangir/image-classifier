from numpy import e
import tensorflow as tf
from tensorflow.keras import layers
from abcli import *
from . import augmentation
from abcli import logging
import logging

logger = logging.getLogger(__name__)


def create_base_model(
    data_object,
    log_level=log_level,
    plot_level=plot_level,
):
    input_shape = (224, 224, 3)
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_layer")

    data_augmentation = augmentation.create_layer(
        data_object=data_object,
        log_level=log_level - 1,
        plot_level=plot_level - 1,
    )
    x = data_augmentation(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

    outputs = layers.Dense(15, activation="softmax", name="output_layer")(x)

    return base_model
