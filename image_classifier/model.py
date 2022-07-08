import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import random
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
        data_path,
        log_level=log_level,
        plot_level=plot_level,
    ):
        """constructor.

        Args:
            data_path (str, optional): path to data. Defaults to "".
        """
        self.log_level = log_level
        self.plot_level = plot_level

        data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, "train"),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        self.input_shape = (224, 224, 3)

        self.base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        self.base_model.trainable = False

        self.inputs = layers.Input(shape=self.input_shape, name="input_layer")

        augmentation_layer = augmentation.create_layer(
            data_path=data_path,
            log_level=log_level - 1,
            plot_level=plot_level - 1,
        )
        x = augmentation_layer(self.inputs)

        x = self.base_model(x, training=False)

        x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

        self.outputs = layers.Dense(
            len(data.class_names),
            activation="softmax",
            name="output_layer",
        )(x)

        self.model = tf.keras.Model(self.inputs, self.outputs)

        self.evaluation = None

        self.history = None

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        if log_level >= LOG_ON:
            print(self.model.summary())

    def evaluate(
        self,
        data_path,
        test_set="test",
    ):
        self.evaluation = self.model.evaluate(
            tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(data_path, test_set),
                image_size=(224, 224),
                label_mode="categorical",
                seed=42,
            )
        )

    def fine_tune(
        self,
        data_path,
        model_path,
        train_set="train",
        validation_set="validation",
        test_set="test",
        unfreeze_layer_count=10,
        epochs=5,
        initial_epoch=1,
        learning_rate_drop_factor=0.1,
        evaluate=True,
    ):
        """fine tune self.

        Args:
            input_object (_type_): _description_
            output_object (_type_): _description_
            unfreeze_layer_count (int, optional): number of layers to unfreeze. Defaults to 10.
            epochs (int, optional): _description_. Defaults to 5.
            train_set (str, optional): _description_. Defaults to "train".
            learning_rate_drop_factor (int, optional): drop factor for learning rate. Defaults to 0.1.
        """
        self.base_model.trainable = True

        for layer in self.base_model.layers[:-unfreeze_layer_count]:
            layer.trainable = False

        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(0.001 * learning_rate_drop_factor),
            metrics=["accuracy"],
        )

        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, train_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, validation_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        # Check which layers are trainable
        # for layer_number, layer in enumerate(model_2.layers[2].layers):
        #    print(layer_number, layer.name, layer.trainable)

        self.history = self.model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=len(train_data),
            validation_data=validation_data,
            validation_steps=len(validation_data),
            initial_epoch=initial_epoch,
        )

        if evaluate:
            self.evaluate(data_path=data_path, test_set=test_set)

    def fit(
        self,
        data_path,
        model_path,
        epochs=5,
        train_set="train",
        validation_set="validation",
        test_set="test",
        log_level=log_level,
        plot_level=plot_level,
        evaluate=True,
    ):
        train_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, train_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )
        validation_data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, validation_set),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        # to speed up - also: callback checkpoints saves don't work w/ load_weight.
        checkpoint_path = os.path.join(model_path, "image_classifier/checkpoint.ckpt")

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
            loss = self.history.history["loss"]
            val_loss = self.history.history["val_loss"]

            accuracy = self.history.history["accuracy"]
            val_accuracy = self.history.history["val_accuracy"]

            epochs = range(len(self.history.history["loss"]))

            # Plot loss
            plt.plot(epochs, loss, label="training_loss")
            plt.plot(epochs, val_loss, label="val_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.legend()

            # Plot accuracy
            plt.figure()
            plt.plot(epochs, accuracy, label="training_accuracy")
            plt.plot(epochs, val_accuracy, label="val_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.legend()

        if evaluate:
            self.evaluate(data_path=data_path, test_set=test_set)

    def load_weights(
        self,
        data_path,
        model_path,
        test_set="test",
        evaluate=False,
    ):
        self.model.load_weights(
            os.path.join(model_path, "image_classifier/checkpoint.ckpt")
        )

        if evaluate:
            self.evaluate(data_path=data_path, test_set=test_set)

    def predict_random_image(
        self,
        data_path,
        log_level=log_level,
        plot_level=plot_level,
    ):
        data = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(data_path, "train"),
            image_size=(224, 224),
            label_mode="categorical",
            seed=42,
        )

        target_class = random.choice(data.class_names)
        target_dir = os.path.join(data_path, "train", target_class)

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
