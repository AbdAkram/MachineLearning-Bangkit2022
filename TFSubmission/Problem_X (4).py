# Given two arrays, train a neural network model to match the X to the Y.
# Desired loss (MSE) < 1e-4

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def solution_A1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                 12.0, 13.0, 14.0, ], dtype=float)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            min_delta=1e-4,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='mymodel.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(X, Y, epochs=500)

    # YOUR CODE HERE

    print(model.predict([-2.0, 10.0]))
    return model
