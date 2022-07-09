# Given two arrays, train a neural network model to match the X to the Y.
# Desired loss (MSE) < 1e-4

import numpy as np
import tensorflow as tf
from tensorflow import keras


def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # YOUR CODE HERE
    normalizer = tf.keras.layers.Normalization(axis=None, input_shape=(1,), name="normalizer")
    normalizer.adapt(X)

    model = tf.keras.Sequential()
    model.add(normalizer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    model.fit(X, Y, epochs=1000, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)])

    # Evaluate the model and check if the loss is less than 1e-4.
    loss, mse = model.evaluate(X, Y)
    assert mse < 1e-4, "\033[91mDesired loss not achieved! Got {:.6f} MSE.\033[0m".format(loss)

    print(model.predict([-2.0, 10.0]))
    return model