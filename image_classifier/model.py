from numpy import e
import tensorflow as tf
from tensorflow.keras import layers
from abcli import *
from . import augmentation
from abcli import logging
import logging

logger = logging.getLogger(__name__)


class Image_Classifier(object):
    def __init__(
        self,
        data_object,
        log_level=log_level,
        plot_level=plot_level,
    ):
        self.log_level = log_level
        self.plot_level = plot_level

        self.input_shape = (224, 224, 3)

        self.base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        self.base_model.trainable = False

        self.inputs = layers.Input(shape=self.input_shape, name="input_layer")

        augmentation_layer = augmentation.create_layer(
            data_object=data_object,
            log_level=log_level - 1,
            plot_level=plot_level - 1,
        )
        x = augmentation_layer(self.inputs)

        x = self.base_model(x, training=False)

        x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

        self.outputs = layers.Dense(15, activation="softmax", name="output_layer")(x)
