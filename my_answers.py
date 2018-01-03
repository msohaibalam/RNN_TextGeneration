import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
import keras

import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i, x in enumerate(series):
        if i + window_size > len(series) - 1:
            break
        else:
            X_tmp_arr = series[i: i + window_size]
            X.append(X_tmp_arr)
            y_tmp_arr = series[i + window_size]
            y.append(y_tmp_arr)

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # initialize the model
    model = Sequential()
    # add LSTM layer with 5 hidden units
    model.add(LSTM(units=5, input_shape=(window_size, 1)))
    # add a fully connected module with 1 unit, with no activation
    model.add(Dense(1))
    # return the model
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    # punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    punctuation = ['!', ',', '.', ':', ';', '?']
    # # also allow numerals
    # numerals = [str(i) for i in range(10)]

    # # replace every character that is not ascii lowercase, or a puncutation character,
    # # or a blank space, or a numeral
    # for ch in list(set(text) - set(string.ascii_lowercase) - set(punctuation) - set(numerals)):
    #     text = text.replace(ch, '')

    # replace every character that is not ascii lowercase, or the characters identified in the list "puncutation"
    for ch in list(set(text) - set(string.ascii_lowercase) - set(punctuation)):
        text = text.replace(ch, ' ')
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    list_text = list(text)

    for i, _ in enumerate(list_text):
        if i + window_size > len(list_text) - 1:
            break
        elif i % step_size == 0:
            inputs_tmp_arr = list_text[i: i + window_size]
            inputs.append(''.join(inputs_tmp_arr))
            outputs_tmp_arr = list_text[i + window_size]
            outputs.append(outputs_tmp_arr)
        else:
            pass

    # # reshape each 
    # inputs = np.asarray(inputs)
    # inputs.shape = (np.shape(inputs)[0:2])
    # outputs = np.asarray(outputs)
    # outputs.shape = (len(outputs),)

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # initialize the model
    model = Sequential()
    # add LSTM layer with 200 hidden units
    model.add(LSTM(units=200, input_shape=(window_size, num_chars)))
    # add Dense layer
    model.add(Dense(units=num_chars, activation='softmax'))
    # return the model
    return model
