# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:35:18 2019

@author: kuangen
"""
from utils import *
from net_model import *
import numpy as np

def test_model_voting(model_type, r_vec = range(5), c_vec = range(16), 
                      delay = 1):
    model, filepath = build_model(model_type = model_type)
    model.load_weights(filepath)
    acc_mat = np.zeros((5,16))
    win_length = 15
    for r in r_vec:
        for c in c_vec:
            exe_str = '_r_' + str(r+1) + '_c_' + str(c+1)
            x_signals, y_signals = load_h5('3-save_data/score_vec.h5', 
                                     data_name = '/data' + exe_str,
                                     label_name = '/label' + exe_str)
            if 0 == len(x_signals) * len(y_signals):
                continue
            x_test, y_test = seg_signals(x_signals, y_signals, 
                                         win_length = win_length)
            y_predict = np.argmax(model.predict(x_test), axis = -1)
            y_predict_filt = voting_filt1(y_predict, filt_delay = delay - 1)
            decisions = np.argmax(y_signals, axis = -1)
            decisions[win_length-1:] = y_predict_filt
            y_test_num = np.argmax(y_test, axis = -1)
            acc_mat[r,c] = calc_acc(y_predict_filt, y_test_num)
    print('Test accuracy:', np.mean(acc_mat[acc_mat != 0]))
    return acc_mat, decisions+1

def model_predict(model_type, r_vec = range(5), c_vec = range(16)):
    model, filepath = build_model(model_type = model_type)
    model.load_weights(filepath)
    y_predict_num_mat = np.empty((5, 16), dtype = np.object)
    y_test_num_mat = np.empty((5, 16), dtype = np.object)
    win_length = 15
    for r in r_vec:
        for c in c_vec:
            exe_str = '_r_' + str(r+1) + '_c_' + str(c+1)
            x_signals, y_signals = load_h5('3-save_data/score_vec.h5', 
                                     data_name = '/data' + exe_str,
                                     label_name = '/label' + exe_str)
            if 0 == len(x_signals) * len(y_signals):
                continue
            x_test, y_test = seg_signals(x_signals, y_signals, 
                                         win_length = win_length)
            y_predict_num_mat[r, c] = np.argmax(model.predict(x_test), axis = -1)
            y_test_num_mat[r,c] = np.argmax(y_test, axis = -1)
    return y_predict_num_mat, y_test_num_mat

def calc_voting_acc(y_predict_num_mat, y_test_num_mat, delay = 1):
    mat_size = y_predict_num_mat.shape
    acc_mat = np.zeros(mat_size)
    for r in range(mat_size[0]):
        for c in range(mat_size[1]):
            if y_predict_num_mat[r,c] is None or y_test_num_mat[r,c] is None:
                continue
            y_predict = y_predict_num_mat[r,c].astype(np.int64)
            y_test = y_test_num_mat[r,c].astype(np.int64)
            y_predict_filt = voting_filt1(y_predict, filt_delay = delay - 1)
            acc_mat[r,c] = calc_acc(y_predict_filt, y_test)
    return acc_mat


#%% test all
#model_type_vec = ['rnn','lstm','gru']
#acc = {}
#for model_type in model_type_vec:
#    model, filepath = build_model(model_type = model_type)
#    acc[model_type] = test_model(model, filepath)
#%% test specific subject
#model_type_vec = ['rnn','lstm','gru']
#acc = {}
#decisions = {}
#for model_type in model_type_vec:
#    acc[model_type], decisions[model_type] = test_model_voting(
#            model_type, r_vec=[0], c_vec=[5])
#%% test accuracy for different delays
model_type_vec = ['rnn','lstm','gru']
acc_mean = np.zeros((11,3,2)) 
acc_std = np.zeros((11,3,2))
decisions = {}
for c in range(len(model_type_vec)):
    y_predict_num_mat, y_test_num_mat = model_predict(model_type_vec[c])
    for delay in range(11):
        acc_mat = calc_voting_acc(y_predict_num_mat, y_test_num_mat, 
                                  delay = delay+1)
        for d in range(2):
            acc_mat_d = acc_mat[:,8*d:8*(d+1)]
            print(np.mean(acc_mat_d[acc_mat_d != 0]))
            acc_mean[delay, c, d] = np.mean(acc_mat_d[acc_mat_d != 0])
            acc_std[delay, c, d] = np.std(acc_mat_d[acc_mat_d != 0])

    
    
    
    


    