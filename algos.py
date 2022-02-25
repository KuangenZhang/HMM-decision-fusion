import time

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


def test_hmm():
    acc_mat = np.zeros((5, 16)) # [trial_num, indoor subject number + outdoor subject number]
    confusion_matrix_mat = np.zeros((5, 16, 5, 5))
    confusion_matrix_CNN_mat = np.zeros((5, 16, 5, 5))

    for r in range(5):
        for c in range(16):
            exe_str = '_r_' + str(r + 1) + '_c_' + str(c + 1)
            x_test, y_test = load_h5('3-save_data/score_vec.h5',
                                     data_name='/data' + exe_str,
                                     label_name='/label' + exe_str,
                                     is_to_categorical=False)
            if 0 == len(x_test) * len(y_test):
                continue
            confusion_matrix_CNN_mat[r, c] = confusion_matrix(y_test.squeeze(), np.argmax(x_test, axis=1))
            acc_mat[r, c], confusion_matrix_mat[r, c] = evaluate_hmm(x_test, y_test)

    acc_mean = np.mean(acc_mat[acc_mat != 0])
    plot_all_confusion_matrix(confusion_matrix_CNN_mat, method='CNN')
    plot_all_confusion_matrix(confusion_matrix_mat, method='HMM')
    return acc_mat, acc_mean

def plot_all_confusion_matrix(confusion_matrix, method = 'HMM'):
    labels = ['LG', 'US', 'DS', 'UR', 'DR']
    plot_confusion_matrix(np.sum(confusion_matrix[:, :8], axis=(0, 1)), labels=labels,
                          title=method + ': Indoor confusion matrix')
    plot_confusion_matrix(np.sum(confusion_matrix[:, 8:], axis=(0, 1)), labels=labels,
                          title=method + ': Outdoor confusion matrix')
    plt.show()

def plot_confusion_matrix(array, labels, title):
    plt.figure()
    n_class = len(labels)
    acc = np.trace(array) / np.sum(array)
    array = array/np.sum(array, axis=1, keepdims=True)
    array = np.round(array * 100, decimals=1)
    sn.heatmap(array, annot=True, cmap='Blues')  # font size
    plt.xticks(np.arange(n_class) + 0.5, labels)
    plt.yticks(np.arange(n_class) + 0.5, labels)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')

    plt.title(title+', accuracy={}%'.format(np.round(100 * acc, decimals=1)))
    plt.savefig('1-results/images/{}.pdf'.format(title), bbox_inches='tight')



def hmm_fusion(score_vec, transit_prob, win_len = 5):
    len_data = score_vec.shape[0]
    hmm_states = np.argmax(score_vec, axis=-1)
    start = time.time()
    for i in range(win_len, len_data - win_len):
        pre_ave_prob = np.mean(score_vec[i - win_len:i, :], axis=0, keepdims=True).reshape((-1, 1))
        post_last_prob = pre_ave_prob * (np.matmul(transit_prob, score_vec[i,:].reshape((-1, 1))))
        last_arg_max = np.argmax(post_last_prob.squeeze())
        hmm_states[i-1] = last_arg_max
        score_vec[i, :] = pre_ave_prob[last_arg_max, 0] * transit_prob[last_arg_max, :] * score_vec[i, :]
        score_vec[i, :] = score_vec[i, :] / np.sum(score_vec[i,:])
        hmm_states[i] = np.argmax(score_vec[i, :])
    print('Calculating time of HMM for each frame (s): ', (time.time()-start) / len_data)
    hmm_states_online = hmm_states.copy()
    hmm_states_online[1:] = hmm_states[0: -1]
    return hmm_states, hmm_states_online


def evaluate_hmm(x_test, y_test):
    transit_prob = np.array([
        [0.80, 0.05, 0.05, 0.05, 0.05],
        [0.15, 0.80, 0.01, 0.03, 0.01],
        [0.15, 0.01, 0.80, 0.01, 0.03],
        [0.15, 0.03, 0.01, 0.80, 0.01],
        [0.15, 0.01, 0.03, 0.01, 0.80]
    ])
    y_predict, _ = hmm_fusion(x_test, transit_prob, win_len=5)
    y_test = y_test.squeeze()
    acc = np.mean(y_predict == y_test)
    confusion_matrix_val = confusion_matrix(
        y_test, y_predict)
    return acc, confusion_matrix_val

