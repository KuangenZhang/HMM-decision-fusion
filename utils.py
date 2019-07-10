# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:22:47 2019

@author: kuangen
"""

import h5py


def load_dataset(folder):
    filename_vec = ['training_set.h5', 'validataion_set.h5' ,'test_set.h5']
    x = []
    y = []
    for filename in filename_vec:
        f = h5py.File(folder + filename, 'r')
        # List all groups
        print("Keys: %s" % f.keys())
        
        x.append(f.get('/data').value)
        y.append(f.get('/label').value)
    return x, y
