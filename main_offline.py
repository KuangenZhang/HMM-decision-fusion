# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:25:07 2019

@author: kuangen
"""
from utils import *
from net_model import *
import algos
#%% load data
print('Loading data...')
x, y = load_dataset('3-save_data/')
print('x_train shape:', x[0].shape, 'y_train shape:', y[0].shape)
#%% train and validate
model_type_vec = ['rnn','lstm','gru']
acc_mean = {}
for model_type in model_type_vec:
    model, filepath = build_model(model_type = model_type)
#    model = train_model(model, x, y, filepath, model_type = model_type)
    _, acc_mean[model_type] = test_model(model, filepath, is_test_speed=False)

# need to transfer the matlab to the python
acc_mean['hmm'] = algos.hmm_fusion()
print(acc_mean)


