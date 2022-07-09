# Build and train a classifier for the sarcasm dataset.
# Desired accuracy and validation_accuracy > 75%

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or test may fail
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json') as f:
        data = json.load(f)
        for item in data:
            sentences.append(item['headline'])
            labels.append(item['is_sarcastic'])

    # Fit your tokenizer with training data
    tokenizer =  Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences) # YOUR CODE HERE

    # Tokenize the training data
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Split the data into training and validation sets
    training_sentences = padded[:training_size]
    training_labels = labels[:training_size]
    validation_sentences = padded[training_size:]
    validation_labels = labels[training_size:]

    # Cast to numpy arrays
    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)

    model = tf.keras.Sequential([
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(training_sentences, training_labels, epochs=10, validation_data=(validation_sentences, validation_labels),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, mode='max',
                                                              factor=0.5)])

    # Predict the validation set
    predictions = model.predict(validation_sentences)

    # Compare the predictions to the validation labels
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] > 0.5 and validation_labels[i] == 1:
            correct += 1
        elif predictions[i] <= 0.5 and validation_labels[i] == 0:
            correct += 1

    # Print the correct count per total count
    print("\033[92mAccuracy: {:.2f}%\033[0m".format(correct / len(predictions) * 100))

    return model
