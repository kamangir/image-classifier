import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from helper_functions import plot_loss_curves
from abcli import *
from . import augmentation
from abcli import logging
import logging

logger = logging.getLogger(__name__)


class Image_Classifier(object):
    def __init__(
        self,
        input_object,
        log_level=log_level,
        plot_level=plot_level,
    ):
        """constructor.

        Args:
            input_object (str, optional): data object. Defaults to "".
        """
        self.log_level = log_level
        self.plot_level = plot_level

        self.input_shape = (224, 224, 3)

        self.base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        self.base_model.trainable = False

        self.inputs = layers.Input(shape=self.input_shape, name="input_layer")

        augmentation_layer = augmentation.create_layer(
            input_object=input_object,
            log_level=log_level - 1,
            plot_level=plot_level - 1,
        )
        x = augmentation_layer(self.inputs)

        x = self.base_model(x, training=False)

        x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

        self.outputs = layers.Dense(15, activation="softmax", name="output_layer")(x)

        self.model = tf.keras.Model(self.inputs, self.outputs)

        self.evaluation = None

        if log_level >= LOG_ON:
            print(self.model.summary())

    def evaluate(
        self,
        input_object,
        test_set="test",
    ):
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        test_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(input_object, test_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        self.evaluation = self.model.evaluate(test_data)

    def fit(
        self,
        input_object,
        output_object,
        epochs=5,
        train_set="train",
        validation_set="validation",
        test_set="test",
        log_level=log_level,
        plot_level=plot_level,
        evaluate=True,
    ):
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(input_object, train_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(input_object, validation_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        # to speed up - also: callback checkpoints saves don't work w/ load_weight.
        checkpoint_path = os.path.join(
            output_object, "image_classifier/checkpoint.ckpt"
        )

        # checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #    filepath=checkpoint_path,
        #    save_weights=True,
        #    save_best_only=True,
        #    save_freq="epoch",
        #    verbose=1,
        # )

        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=len(train_data),
            validation_data=validation_data,
            validation_steps=len(validation_data),
            # callbacks=[checkpoint_callback],
        )

        self.model.save_weights(checkpoint_path)

        if plot_level >= PLOT_ON:
            plot_loss_curves(self.history)

        if evaluate:
            self.evaluate(input_object=input_object, test_set=test_set)

    def load_weights(
        self,
        input_object,
        output_object,
        test_set="test",
        evaluate=False,
    ):
        self.model.load_weights(
            os.path.join(output_object, "image_classifier/checkpoint.ckpt")
        )

        if evaluate:
            self.evaluate(input_object=input_object, test_set=test_set)

    def predict_random_image(
        self,
        input_object,
        log_level=log_level,
        plot_level=plot_level,
    ):
        data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(input_object, "train"),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        target_class = random.choice(data.class_names)
        target_dir = os.path.join(input_object, "train", target_class)

        image = mpimg.imread(
            os.path.join(target_dir, random.choice(os.listdir(target_dir)))
        )

        confidence = tf.squeeze(
            self.model(
                tf.expand_dims(image, axis=0),
                training=True,
            )
        )
        index = np.argmax(confidence)

        message = f"{target_class} -model-> {data.class_names[index]} ({confidence[index]:.2f})"

        if plot_level >= PLOT_ON:
            logger.info(message)

        if log_level >= LOG_ON:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title(message)
            plt.axis(False)
            plt.show()
