# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:25:07 2019

@author: kuangen
"""

'''
#Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
**Notes**
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical
from utils import *
import numpy as np

num_classes = 5
sequence_len = 15
feature_len = 5
batch_size = 64

print('Loading data...')
# [training, validation, test]
x, y = load_dataset('3-save_data/')
for i in range(len(y)):
    y[i] = y[i] - np.min(y[i])
    y[i] = to_categorical(y[i], num_classes= num_classes)
print('x_train shape:', x[0].shape)
print('x_test shape:', x[2].shape)

print('Build model...')
model = Sequential()
model.add(LSTM(units = 128, dropout=0.2, recurrent_dropout=0.2, 
               input_shape=(sequence_len, feature_len)))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x[0], y[0],
          batch_size=batch_size,
          epochs=15,
          validation_data=(x[1], y[1]))
score, acc = model.evaluate(x[2], y[2],
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)