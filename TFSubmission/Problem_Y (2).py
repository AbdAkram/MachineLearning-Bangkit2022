# Build a classifier for the Fashion MNIST dataset.
# Desired accuracy AND validation_accuracy > 83%

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # DEFINE YOUR MODEL HERE
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='model_B2.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # End with 10 Neuron Dense, activated by softmax
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # COMPILE MODEL HERE
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        training_images,
        training_labels,
        batch_size=128,
        epochs=20,
        verbose=1,
        validation_data=(test_images, test_labels),
        callbacks=callbacks
    )

    # TRAIN YOUR MODEL HERE
    return model
