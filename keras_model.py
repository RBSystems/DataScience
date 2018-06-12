from keras import Input, layers
import numpy as np

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
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# Coppied from Deep Learning with Python by Francois Chollet
def normalize_data(float_data):
    mean = float_data.mean(axis=0)
    float_data -= mean
    std = float_data.std(axis=0)
    float_data /= std
    return float_data