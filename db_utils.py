import numpy
import os
import scipy.io.wavfile
import python_speech_features
import re
import wave
import matplotlib.pyplot as plt
import numpy as np

################################################################################
######Google Speech Commands Dataset for Speech Recognition#####################
################################################################################

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class DataFetcher:

    def __init__(self, path, classes, cepstrum_dimension,
     winlen=0.020, winstep=0.01, encoder_filter_frames= 8, encoder_stride= 5):
        self.path = path
        self.classes = classes
        self.cepstrum_dimension = cepstrum_dimension
        self.winlen = winlen
        self.winstep = winstep
        self.recording = False
        self.frames = numpy.array([], dtype=numpy.int16)
        self.samplerate = 16000
        self.encoder_stride = encoder_stride
        self.encoder_filter_frames = encoder_filter_frames
        self.train_data = {}
        self.validation_data = {}
        self.test_data = {}

        for sound_class in self.classes:
            self.validation_data[sound_class] = set()
            self.test_data[sound_class] = set()

        # Parsing the validation set
        validation_path = path + '/training_settings/validation_list.txt'
        for line in open(validation_path, 'r'):
            class_name = re.search('^[a-z|_]*', line).group(0)
            if class_name in self.classes:
                file_name = re.search('([a-z]|[0-9]|_)*.wav', line).group(0)
                self.validation_data[class_name].add(file_name)

        # Parsing the test set
        test_path = path + '/training_settings/testing_list.txt'
        for line in open(test_path, 'r'):
            class_name = re.search('^[a-z|_]*', line).group(0)
            if class_name in self.classes:
                file_name = re.search('([a-z]|[0-9]|_)*.wav', line).group(0)
                self.test_data[class_name].add(file_name)

        # Parsing the trainning set as all minus validation and test
        for sound_class in classes:
            wav_files = os.listdir(self.path+'/'+sound_class)
            self.train_data[sound_class] = set(wav_files).difference(self.validation_data[sound_class].union(self.test_data[sound_class]))

        for name_class in classes:
            if (len(self.test_data[name_class]) < 2):
                print("missing samples for", name_class, "test")
            if (len(self.validation_data[name_class]) < 2):
                print("missing samples for", name_class, "validation")
            if (len(self.train_data[name_class]) < 2):
                print("missing samples for", name_class, "trainning")


    def _get_set(self, origin, class_parameter):
        """Get set according to origin and class."""
        if origin == "training":
            return self.train_data[class_parameter]
        elif origin == "validation":
            return self.validation_data[class_parameter]
        elif origin == "test":
            return self.test_data[class_parameter]
        else:
            return False

    # This function returns a single strided mfcc from the selected class
    def get_mcc_from_class(self, class_parameter, num=10000, origin="training"):
        # wav_file = numpy.random.choice(list(self._get_set(origin, class_parameter)), size=1, replace=False)[0]
        wav_files = self._get_set(origin, class_parameter)
        max_size = 99
        mfcc_array = []
        count = 0
        for f in wav_files:
            if count<num:
                (rate,sig) = scipy.io.wavfile.read(self.path+'/'+class_parameter+'/'+f)
                mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=512, lowfreq=0, highfreq=16000/2)

                if mfcc.shape[0]!= max_size:
                    mfcc = np.pad(mfcc, ((max_size - mfcc.shape[0], 0),(0,0)), mode='constant')
                mfcc_array.append(mfcc)
            else:
                break
            count += 1
        return mfcc_array

    # This function
    def get_mfcc_from_file(self, wav_file):
        """Return a single strided mfcc from a file."""
        (rate,sig) = scipy.io.wavfile.read(wav_file)
        mfcc = python_speech_features.mfcc(sig, rate, winlen=0.020, winstep=0.01, numcep=self.cepstrum_dimension, nfilt=self.cepstrum_dimension, nfft=512, lowfreq=0, highfreq=16000/2)
        return self._mfcc_to_strided(mfcc)

def get_speech_dataset(dir=DIR_PATH):
    """Get full digit dataset, divided in train, test and validation sets."""

    path = dir+'/speech_commands_v0.01'
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    cepstrum_dimension= 28
    audio_data = DataFetcher(path, classes, cepstrum_dimension)

    # Initialize Variables
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    num = 0

    for c in classes:

        # Training Data
        train_step = audio_data.get_mcc_from_class(c)
        X_train = X_train + train_step
        y_train = y_train + [num]*len(train_step)

        # Test Data
        test_step = audio_data.get_mcc_from_class(c, origin="test")
        X_test = X_test + test_step
        y_test = y_test + [num]*len(test_step)

        # Validation Data
        val_step = audio_data.get_mcc_from_class(c, origin="validation")
        X_val = X_val + val_step
        y_val = y_val + [num]*len(val_step)

        num += 1

    # Transform lists into arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def get_speech_dataset_complete(dir=DIR_PATH):
    """Get full digit dataset, divided in train, test and validation sets."""

    path = dir+'/speech_commands_v0.01'
    classes = ['bed', 'bird', 'cat', 'dog', 'down', 'go','happy', 'house', 'left', 'marvin', 'no', 'off', 'on', 'right', 'sheila', 'stop', 'tree', 'up', 'wow', 'yes','zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    cepstrum_dimension= 50
    audio_data = DataFetcher(path, classes, cepstrum_dimension)

    # Initialize Variables
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    num = 0

    for c in classes:

        # Training Data
        train_step = audio_data.get_mcc_from_class(c)
        X_train = X_train + [train_step, train_step, train_step]
        y_train = y_train + [num]*len(train_step)

        # Test Data
        test_step = audio_data.get_mcc_from_class(c, origin="test")
        X_test = X_test + [test_step, test_step, test_step]
        y_test = y_test + [num]*len(test_step)

        # Validation Data
        val_step = audio_data.get_mcc_from_class(c, origin="validation")
        X_val = X_val + [val_step, val_step, val_step]
        y_val = y_val + [num]*len(val_step)

        num += 1

    # Transform lists into arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

def get_single_digit_dataset(digit, dir=DIR_PATH):
    """Get full digit dataset, divided in train, test and validation sets."""

    path = dir+'/speech_commands_v0.01'
    classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    cepstrum_dimension= 28
    audio_data = DataFetcher(path, classes, cepstrum_dimension)

    # Initialize Variables
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []

    num = 0

    for c in classes:

        if c == classes[digit]:
            # Training Data
            train_step = audio_data.get_mcc_from_class(c)
            X_train = X_train + train_step
            y_train = y_train + [num]*len(train_step)
            # Test Data
            test_step = audio_data.get_mcc_from_class(c, origin="test")
            X_test = X_test + test_step
            y_test = y_test + [num]*len(test_step)
            # Validation Data
            val_step = audio_data.get_mcc_from_class(c, origin="validation")
            X_val = X_val + val_step
            y_val = y_val + [num]*len(val_step)
        else:
            # Training Data
            train_step = audio_data.get_mcc_from_class(c, num=180)
            X_train = X_train + train_step
            y_train = y_train + [num]*len(train_step)
            # Test Data
            test_step = audio_data.get_mcc_from_class(c, num=25, origin="test")
            X_test = X_test + test_step
            y_test = y_test + [num]*len(test_step)
            # Validation Data
            val_step = audio_data.get_mcc_from_class(c, num=25, origin="validation")
            X_val = X_val + val_step
            y_val = y_val + [num]*len(val_step)

        num += 1

    # Transform lists into arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    y_val = np.asarray(y_val)

    return (X_train, y_train), (X_test, y_test), (X_val, y_val)

################################################################################
######IMDB Dataset for Natural Language Processing##############################
################################################################################

import os
import re
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def import_files(folder):
    """Import file contents from IMDb folders."""
    file_names = os.listdir(folder)
    dataset = []
    for file in file_names:
        file = folder + file
        with open(file, "r") as f:
            dataset.append(f.read())
    return dataset

def clean_dataset(dataset):
    """Clean dataset of html tags, upper case and other symbols."""
    step_1 = re.compile("[.;:!\'?,\"()\[\]]")
    clean_dataset = [step_1.sub("", line.lower()) for line in dataset]
    step_2 = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    clean_dataset = [step_2.sub(" ", line) for line in clean_dataset]
    return clean_dataset

def vectorize_dataset(dataset):
    """Vectorize dataset from set vocabulary."""

    vectorizer = CountVectorizer(binary=True)

    # Obtain Vocabulary from file
    vocabulary = []
    with open(DIR_PATH+"/imdb/imdb.vocab", "r") as f:
        vocabulary = f.read().splitlines()

    # Fit and apply vectorizer to dataset
    vectorizer.fit(vocabulary)
    vectorized_dataset = vectorizer.transform(dataset)
    vectorized_dataset_dense = vectorized_dataset.todense()

    return vectorized_dataset

def get_imdb_dataset(dir=DIR_PATH):
    """Import whole train and test dataset for IMDb reviews."""
    # Import X_train
    dir = dir + "/imdb"
    X_train = import_files(dir+"/train/pos/")
    train_positive_items = len(X_train)
    X_train = np.asarray(X_train)
    X_train = np.concatenate((X_train, np.asarray(import_files(dir+"/train/neg/"))))
    train_negative_items = len(X_train) - train_positive_items
    # Import X_test
    X_test = import_files(dir+"/test/pos/")
    test_positive_items = len(X_test)
    X_test = np.asarray(X_test)
    X_test = np.concatenate((X_test, np.asarray(import_files(dir+"/test/neg/"))))
    test_negative_items = len(X_test) - test_positive_items

    # y_train to Categorical
    y_train = np.ones(train_positive_items)
    train_negative_items = np.zeros(train_negative_items)
    y_train = np.concatenate((y_train, train_negative_items))
    y_train = to_categorical(y_train)

    # y_test to Categorical
    y_test = np.ones(test_positive_items)
    test_negative_items = np.zeros(test_negative_items)
    y_test = np.concatenate((y_test, test_negative_items))
    y_test = to_categorical(y_test)

    # Clean datasets
    X_train = clean_dataset(X_train)
    X_test = clean_dataset(X_test)

    # Vectorize dataset
    X_train = vectorize_dataset(X_train)
    X_test = vectorize_dataset(X_test)

    X_train = X_train.todense()
    X_test = X_test.todense()

    return (X_train, y_train), (X_test, y_test)

def prepare_samples(samples):
    """Prepare sample for model."""
    clean_samples = clean_dataset(samples)
    prepared_samples = vectorize_dataset(clean_samples)

    return prepared_samples


def get_imdb_dataset_v2(dir=DIR_PATH):
    """Import whole train and test dataset for IMDb reviews."""

    df_train = DataFrame()
    df_test = DataFrame()

    # Import X_train
    dir = dir + "/imdb"
    X_train = import_files(dir+"/train/pos/")
    train_positive_items = len(X_train)
    X_train = np.asarray(X_train)
    X_train = np.concatenate((X_train, np.asarray(import_files(dir+"/train/neg/"))))
    train_negative_items = len(X_train) - train_positive_items

    # Import X_test
    X_test = import_files(dir+"/test/pos/")
    test_positive_items = len(X_test)
    X_test = np.asarray(X_test)
    X_test = np.concatenate((X_test, np.asarray(import_files(dir+"/test/neg/"))))
    test_negative_items = len(X_test) - test_positive_items

    # Update dataframe
    df_train['X_raw'] = X_train
    df_test['X_raw'] = X_test

    # y_train to Categorical
    y_train = np.ones(train_positive_items)
    train_negative_items = np.zeros(train_negative_items)
    y_train = np.concatenate((y_train, train_negative_items))
    y_train = to_categorical(y_train)

    # y_test to Categorical
    y_test = np.ones(test_positive_items)
    test_negative_items = np.zeros(test_negative_items)
    y_test = np.concatenate((y_test, test_negative_items))
    y_test = to_categorical(y_test)

    # Update dataframe y
    df_train['y'] = y_train
    df_test['y'] = y_test

    # Clean datasets
    X_train = clean_dataset(X_train)
    X_test = clean_dataset(X_test)

    # Vectorize dataset
    X_train = vectorize_dataset(X_train)
    X_test = vectorize_dataset(X_test)

    # Sparse matrix to dense
    X_train = X_train.todense()
    X_test = X_test.todense()

    # Update dataframe
    df_train['X_clean'] = X_train
    df_test['X_clean'] = X_test

    return df_train, df_test

if __name__ == '__main__':

    test, train = get_imdb_dataset_v2()
