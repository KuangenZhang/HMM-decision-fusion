# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:22:47 2019

@author: kuangen
"""

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy import stats


def load_dataset(folder, num_classes = 5):
    filename_vec = ['training_set.h5', 'validataion_set.h5']
    x = []
    y = []
    for filename in filename_vec:
        f = h5py.File(folder + filename, 'r')
        # List all groups
        print("Keys: %s" % f.keys())
        x.append(f.get('/data').value)
        y.append(y_to_categorical(f.get('/label').value, 
                                  num_classes = num_classes))
    return x, y


def load_h5(file_path, data_name = '/data', label_name = '/label', 
            num_classes = 5, is_to_categorical = True):
    f = h5py.File(file_path, 'r')
    if data_name in f:
        if is_to_categorical:
            return f.get(data_name).value, y_to_categorical(f.get(label_name).value,
                         num_classes = num_classes)
        else:
            # The h5 file is generated from the matlab, where the first index is 1.
            # Here I decrease the label by 1 because the first index in python is 0.
            return f.get(data_name)[...], f.get(label_name)[...] - 1
    else:
        return [], []

def y_to_categorical(y, num_classes = 5):
    y = y - np.min(y)
    y = to_categorical(y, num_classes= num_classes)
    return y

def seg_signals(x_signals, y_signals, win_length = 15):
    y_test = y_signals[win_length-2:-1,:]
    x_size = x_signals.shape
    len_y_test = y_test.shape[0]
    x_test = np.zeros((len_y_test, win_length, x_size[-1]));
    for i in range(len_y_test):
        x_test[i, :, :] = x_signals[i:i+win_length, :]
    return x_test, y_test

def voting_filt1(y, filt_delay = 1):
    y_filt = np.copy(y)    
    for i in range(filt_delay,len(y) - filt_delay):
        y_filt[i] = stats.mode(y[(i-filt_delay):(i+filt_delay+1)]).mode
    return y_filt

def calc_acc(y_predict, y_correct):
    return (y_predict == y_correct).mean()
    
    