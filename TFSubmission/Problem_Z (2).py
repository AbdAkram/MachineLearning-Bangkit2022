# Create a classifier for the MNIST Handwritten digit dataset.
# Desired accuracy AND validation_accuracy > 91%

import tensorflow as tf


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # NORMALIZE YOUR IMAGE HERE
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # DEFINE YOUR MODEL HERE
    # End with 10 Neuron Dense, activated by softmax
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # COMPILE MODEL HERE
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TRAIN YOUR MODEL HERE
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)])

    return model
