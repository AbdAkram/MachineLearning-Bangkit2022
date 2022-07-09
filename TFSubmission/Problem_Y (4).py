# Build and train a classifier for the BBC-text dataset.
# Desired accuracy and validation_accuracy > 91%

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    sentences = []
    X = []
    y = []

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or  you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    train_data, test_data = train_test_split(bbc, train_size=training_portion, shuffle=False)

    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_data['text'])

    # Transform training data and test data
    train_x = tokenizer.texts_to_sequences(train_data['text'])
    test_x = tokenizer.texts_to_sequences(test_data['text'])

    # Pad the sequences to the same length
    train_x = pad_sequences(train_x, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    test_x = pad_sequences(test_x, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # List all the unique labels
    labels = sorted(list(set(train_data['category'])))

    # Convert labels to index
    train_y = np.array([labels.index(x) for x in train_data['category']])
    test_y = np.array([labels.index(x) for x in test_data['category']])

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_x, train_y, epochs=100, callbacks=[], validation_data=(test_x, test_y))

    return model
