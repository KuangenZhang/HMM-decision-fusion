# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:31:14 2019

@author: kuangen
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import *
import numpy as np

def build_model(model_type = 'lstm', num_classes = 5, sequence_len = 15, 
                feature_len = 5):
    units = 128
    dropout = 0.0
    recurrent_dropout = 0.0
    print('Build model...')
    model = Sequential()
    if 'rnn' == model_type:
        model.add(SimpleRNN(units = units, return_sequences=False,
                            dropout= dropout, 
                            recurrent_dropout= recurrent_dropout,
                            input_shape=(sequence_len, feature_len)))
    elif 'lstm' == model_type:
        model.add(LSTM(units = units, return_sequences=False,
                        dropout= dropout, 
                        recurrent_dropout= recurrent_dropout,
                       input_shape=(sequence_len, feature_len)))
    elif 'gru' == model_type:
        model.add(GRU(units = units, return_sequences=False,
                      dropout= dropout, 
                      recurrent_dropout= recurrent_dropout,
                      input_shape=(sequence_len, feature_len)))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()
    filepath = '4-save_model/' + model_type + '_model.hdf5'
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, filepath

def train_model(model, x, y, filepath, model_type = 'lstm', batch_size = 128):
    print('Train...')
    # checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(x[0], y[0],
              batch_size=batch_size,
              epochs=15,
              validation_data=(x[1], y[1]), callbacks=callbacks_list)
    return model

def test_model(model, filepath, is_test_speed = False):
    model.load_weights(filepath)
    acc_mat = np.zeros((5, 16))
    if is_test_speed:
        r = 0
        c = 1
        exe_str = '_r_' + str(r + 1) + '_c_' + str(c + 1)
        x_test, y_test = load_h5('3-save_data/subject_test_set.h5',
                                 data_name='/data' + exe_str,
                                 label_name='/label' + exe_str)
        _, acc_mat[r, c] = model.evaluate(x_test, y_test, batch_size=1)
    else:
        for r in range(5):
            for c in range(16):
                exe_str = '_r_' + str(r+1) + '_c_' + str(c+1)
                x_test, y_test = load_h5('3-save_data/subject_test_set.h5',
                                         data_name = '/data' + exe_str,
                                         label_name = '/label' + exe_str)
                if 0 == len(x_test) * len(y_test):
                    continue
                _, acc_mat[r, c] = model.evaluate(x_test, y_test)
        acc_mean = np.mean(acc_mat[acc_mat != 0])
        print('Test accuracy:', acc_mean)
    return acc_mat, acc_mean