# Given two arrays, train a neural network model to match the X to the Y.
# Desired loss (MSE) < 1e-3

import numpy as np
import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 1e-3):
            self.model.stop_training = True


def solution_B1():
    # DO NOT CHANGE THIS CODE
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    Y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    layer_norm = keras.layers.Normalization(input_shape=[1], axis=None)
    layer_norm.adapt(X)

    model = tf.keras.models.Sequential([
        layer_norm,
        keras.layers.Dense(1)
    ])

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.SGD(learning_rate=0.5)
                  )

    CALLBACK = myCallback()

    model.fit(X, Y, epochs=100, callbacks=[CALLBACK])
    # YOUR CODE HERE

    print(model.predict([-2.0, 10.0]))
    return model
