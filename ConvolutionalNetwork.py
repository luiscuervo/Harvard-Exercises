import os
import sys
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

from matplotlib import pyplot as plt

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)
from db_utils import get_imdb_dataset, get_speech_dataset

def Secure_Voice_Channel(func):
    """Define Secure_Voice_Channel decorator."""
    def execute_func(*args, **kwargs):
        print('Established Secure Connection.')
        returned_value = func(*args, **kwargs)
        print("Ended Secure Connection.")

        return returned_value

    return execute_func

@Secure_Voice_Channel
def generic_vns_function(input_dim, number_dense_layers, classes, units, lr):
    """Generic Deep Learning Model generator."""
    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size=3, activation= 'relu'))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())

    for i in range(number_dense_layers):
        model.add(layers.Dense(units=units, input_dim= input_dim,
                               kernel_initializer='normal', activation='relu'))

    model.add(layers.Dense(classes, kernel_initializer='normal',
              activation='softmax'))
    opt = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    return model

def train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    """Generic Deep Learning Model training function."""

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
              batch_size=batch_size, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def choose_dataset(dataset_type):
    """Select dataset based on string variable."""
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    elif dataset_type == "computer_vision":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_type == "speech_recognition":
        (X_train, y_train), (X_test, y_test), (_, _) = get_speech_dataset()
    else:
        raise ValueError("Couldn't find dataset.")

    (X_train, X_test) = normalize_dataset(dataset_type, X_train, X_test)

    (X_train, y_train), (X_test, y_test) = reshape_dataset(X_train, y_train, X_test, y_test)

    return (X_train, y_train), (X_test, y_test)

def normalize_dataset(string, X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    if string is "computer vision":
        X_train = X_train / 255
        X_test = X_test / 255
    else:
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train-std)/mean
        X_test = (X_test-std)/mean

    return (X_train, X_test)

def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape Computer Vision and Speech datasets."""

    num_pixels = X_test.shape[1]*X_test.shape[2]
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2], 1).astype('float32')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)

def main():

    # Hyperparameters
    layers = 2
    layer_units = 1000
    epochs = 7
    batch_size = 200

    # Dataset :  "computer_vision" or "speech_recognition"
    dataset = "speech_recognition"

    # Import Datasets
    (X_train, y_train), (X_test, y_test) = choose_dataset(dataset)


    # Generate and train model
    lr = 0.0001
    model = generic_vns_function(X_train.shape[1], layers, y_train.shape[1], layer_units, lr)
    trained_model = train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test)

    # Save model to h5 file
    trained_model.save('models/model_%s_a1.h5' % dataset)

    return None

if __name__ == '__main__':
    main()
