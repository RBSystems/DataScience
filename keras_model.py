import json
import os
import os.path
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from keras.layers import GRU, Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy import nan
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings


def get_test_data():
    df = pd.read_hdf('./checkpoints/data_test_normalized.h5','table')
    return df.values


def get_full_data():
    df = pd.read_hdf('./checkpoints/state_data_full_normalized.h5','table')
    return df.values


def get_truncated_data():
    df = pd.read_hdf('./checkpoints/state_data_truncated_normalized.h5','table')
    return df.values


# Coppied from Deep Learning with Python by Francois Chollet
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    """ Generator yielding timeseries samples and their targets
    
    Arguments:
        data {[type]} -- The original array of floating-point data,
        lookback {[type]} -- How many timesteps back the input data should go.
        delay {[type]} -- How many timesteps in the future the target should be.
        min_index {[type]} -- Indices in the data array that delimit which timesteps to draw from.
        max_index {[type]} -- Indices in the data array that delimit which timesteps to draw from.
    
    Keyword Arguments:
        shuffle {bool} -- Whether to shuffle the samples or draw them in chronological order. (default: {False})
        batch_size {int} -- The number of samples per batch. (default: {128})
        step {int} -- The period, in timesteps, at which you sample data. (default: {6})
    """

    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows), data.shape[-1]))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay]
        yield samples, targets


# Coppied from Deep Learning with Python by Francois Chollet
def normalize_data(float_data):
    mean = float_data.mean(axis=0)
    float_data -= mean
    std = float_data.std(axis=0)
    float_data /= std
    return float_data


def get_train_test_val(float_data, lookback, step, delay, batch_size):
    train_index = int(len(float_data)*(.5))
    test_index = train_index + int(len(float_data)*(.25))

    train_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=0,
                        max_index=train_index,
                        step=step, 
                        batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=train_index + 1,
                        max_index=test_index,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=test_index + 1,
                        max_index=None,
                        step=step,
                        batch_size=batch_size)

    # This is how many steps to draw from `val_gen`
    # in order to see the whole validation set:
    val_steps = (test_index - (train_index + 1) - lookback) // batch_size

    # This is how many steps to draw from `test_gen`
    # in order to see the whole test set:
    test_steps = (len(float_data) - (test_index + 1) - lookback) // batch_size

    return train_gen, val_gen, test_gen, val_steps, test_steps


def get_model(float_data):
    model = Sequential()
    model.add(GRU(32, dropout=0.1,
                        recurrent_dropout=0.3,
                        return_sequences=True,
                        input_shape=(None, float_data.shape[-1])))
    model.add(GRU(64, activation='relu',
                        dropout=0.1, 
                        recurrent_dropout=0.3))
    model.add(Dense(float_data.shape[-1]))
    return model 


def save_model(model, base_name):
    # Save the weights
    weights = base_name + '_weights.h5'
    model.save_weights(weights)

    # Save the model architecture
    arc_name = base_name + '_arc.h5'
    with open(arc_name, 'w') as f:
        f.write(model.to_json())
    
    # Save the model itself
    model.save(base_name + '_model.h5')


def train_model_test(model_save_name):
    float_data = get_test_data()

    lookback = 10080    # 1 week in minutes
    step = 1   
    delay = 240        # 2 hours in minutes
    batch_size = 128

    train_model(model_save_name, float_data, lookback, step, delay, batch_size)


def train_model_full(model_save_name):
    float_data = get_full_data()

    lookback = 1440    # 1 week in minutes
    step = 1   
    delay = 240        # 2 hours in minutes
    batch_size = 128

    train_model(model_save_name, float_data, lookback, step, delay, batch_size)


def train_model_truncated(model_save_name):
    float_data = get_truncated_data()

    lookback = 1008    # 1 week in 10 minutes
    step = 1           # 10 mins
    delay = 144        # 2 hours in minutes
    batch_size = 128

    train_model(model_save_name, float_data, lookback, step, delay, batch_size)



def train_model(model_save_name, float_data, lookback, step, delay, batch_size):
    train_gen, val_gen, test_gen, val_steps, test_steps = get_train_test_val(float_data, lookback, step, delay, batch_size)
    model = get_model(float_data)
    print("loaded model")

    model.compile(optimizer=RMSprop(), loss='mae', metrics=['mae', 'acc'])
    save_name = model_save_name + '_auto.h5'
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, mode='auto')
    checkpoint = ModelCheckpoint(filepath=save_name, monitor='val_loss',save_best_only=True)
    callbacks_list = [earlystop, checkpoint]
    history = model.fit_generator(train_gen,
                                steps_per_epoch=500,
                                epochs=40,
                                callbacks=callbacks_list,
                                validation_data=val_gen,
                                validation_steps=val_steps)
    save_model(model, model_save_name)
    scores = model.evaluate_generator(test_gen, steps=test_steps)
    print("Accuracy = ", scores)
    plot(history, model_save_name)


def plot(history, model_save_name):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    name = model_save_name + '_fig.png'
    plt.savefig(name)


train_model_truncated("two_GRU_trunc")

train_model_test("two_GRU_test")
