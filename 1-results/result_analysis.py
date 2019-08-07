# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


def plot_acc_mean_std_subject(data, image_name, sub_num, legend_vec):
    figurePath = 'images/'
    figName = figurePath + image_name
    plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 15})
    plt.tight_layout()
    accuracy_list = []
    # rows: methods; cols: repeating tests
    for i in range(sub_num):
        data_mat = data[:, i].reshape((-1,5))
        accuracy_list.append(data_mat)
    plot_error_bar_for_cell(accuracy_list, [], 'Classification accuracy (%)', legend_vec)

    sub_name_list = ['S2','S3','S4','S5','A1','A2','A3']
    legend_num = len(legend_vec)
    plt.xlim(0, sub_num * (legend_num +2))
    plt.xticks(np.arange((legend_num + 1) / 2, sub_num * (legend_num +2), (legend_num +2)),
               sub_name_list)
    plt.ylim(0.85, 1)
    plt.yticks(np.arange(0.85, 1.01, 0.05), ['85', '90', '95', '100'])
    plt.savefig(figName + '.pdf', bbox_inches='tight')
    plt.show()

def plot_acc_target_subject(data, image_name, sub_num, legend_vec):
    figurePath = 'images/'
    figName = figurePath + image_name
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    accuracy_list = []
    for i in range(sub_num):
        accuracy_list.append(data[1::2, i])
    
    plot_bar_for_cell(accuracy_list, [], 'Classification accuracy (%)', legend_vec)
    
    sub_name_list = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
    plt.xlim(0, 7 * sub_num)
    plt.xticks(range(3, 3 + 7 * (sub_num), 7), sub_name_list[0:sub_num])
    plt.ylim(0.65, 1)
    plt.yticks(np.arange(0.65,1.05,0.05),['65', '70', '75','80','85','90','95','100'])
    plt.rcParams.update({'font.size': 13})
    plt.savefig(figName + '.pdf', bbox_inches='tight')

def plot_acc_mean_std(data, image_name, sub_num, legend_vec):
    figurePath = 'images/'
    figName = figurePath + image_name
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    accuracy_list = []
    # rows: methods; cols: subjects
    for i in range(2):
        accuracy_list.append(data[i::2, :])
    
    plot_error_bar_for_cell(accuracy_list, [], 'Classification accuracy (%)', legend_vec)
    
    sub_name_list = ['Source subjects','Target subjects']
    plt.xlim(0, np.ceil(data.shape[0] / 4) + data.shape[0])
    plt.xticks([np.ceil(data.shape[0] / 4), data.shape[0]],
                sub_name_list)
    plt.ylim(0.65, 1)
    plt.yticks(np.arange(0.65,1.05,0.05),['65', '70', '75','80','85','90','95','100'])
    plt.rcParams.update({'font.size': 13})
    plt.savefig(figName + '.pdf', bbox_inches='tight')
    
def plot_error_bar_for_cell(input_list, xLabel, yLabel, legend_vec):
    len_list = len(input_list)
    for i in range(len_list):
        input_vec = input_list[i]
        len_cols = len(input_vec)
        x_vec = range(i*(len_cols+2)+1, i*(len_cols+2)+len_cols+1)
        plot_error_bar(x_vec, input_vec, legend_vec)
    plt.ylabel(yLabel)

def plot_bar_for_cell(input_list, xLabel, yLabel, legend_vec):
    len_list = len(input_list)
    for i in range(len_list):
        input_vec = input_list[i]
        len_cols = len(input_vec)
        x_vec = range((i)*(len_cols+2)+1, (i)*(len_cols+2)+len_cols+1)
        plot_bar(x_vec, input_vec, legend_vec)
    plt.ylabel(yLabel)
    

def plot_bar(x_vec, input_vec, legend_vec):
    len_vec = len(x_vec)
    color_vec = cm.get_cmap('Set3', 12)
    hatch_vec = ['', '-', '/', '\\', 'x', '-/','-\\','-x']
    for i in range(len_vec):
        plt.bar(x_vec[i], input_vec[i], color=color_vec(i), edgecolor='black',
                hatch = hatch_vec[i])
    plt.legend(legend_vec, loc = 'lower center', 
               ncol = len(legend_vec), bbox_to_anchor = (0.49, 1.0))
    
def plot_error_bar(x_vec, input_vec, legend_vec):
    mean_vec = np.mean(input_vec, axis = -1)
    std_vec = np.std(input_vec, axis = -1)
    len_vec = len(x_vec)
    color_vec = cm.get_cmap('Set3', 12)
    hatch_vec = ['', '-', '/', '\\', 'x', '-/','-\\','-x']
    for i in range(len_vec):
        plt.bar(x_vec[i], mean_vec[i], color=color_vec(i), edgecolor='black',
                hatch = hatch_vec[i])
    plt.errorbar(x_vec, mean_vec, yerr = std_vec, fmt='.', elinewidth= 1,
                 solid_capstyle='projecting', capsize= 3, color = 'black')
    plt.legend(legend_vec, loc = 'lower center', 
               ncol = 3, bbox_to_anchor = (0.49, 1.0))


def plot_error_line(x_vec, acc_mean_mat, acc_std_mat):
    # acc_mean_mat, acc_std_mat: rows: delays, cols: methods
    fmt_vec = ['o-', '+--', 'v-.', 'x-', 'd-', '*-.']
    color_vec = plt.cm.Dark2(np.arange(8))
    for i in range(acc_mean_mat.shape[1]):
        plt.errorbar(x_vec, acc_mean_mat[:, i], fmt = fmt_vec[i], capsize= 5,
                     elinewidth = 2, markersize = 10, color = color_vec[i])


def plot_line(x_vec, y_mat, only_plot_error = True,
              marker_vec = ['o', '+', 'v', 'x', 'd', '*', ''],
              line_vec = ['', '', '', '', '', '', ''],
              line_width_vec = [2, 2, 2, 2, 2, 2, 2], marker_size = 10,
              is_last_black = True):
    # y_mat: rows: frames, cols: methods
    cols_y = y_mat.shape[1]
    color_vec = plt.cm.Dark2(np.arange(cols_y))
    if is_last_black:
        color_vec[-1, 0:3] = 0
    for i in range(cols_y):
        if only_plot_error and i != (cols_y - 1):
            idx = y_mat[:, i] != y_mat[:, -1]
            plt.plot(x_vec[idx], y_mat[idx, i], linestyle = line_vec[i],
                     marker = marker_vec[i], markersize = marker_size, linewidth = line_width_vec[i],
                     color = color_vec[i])
        else:
            plt.plot(x_vec, y_mat[:, i], linestyle=line_vec[i],
                     marker= marker_vec[i], markersize = marker_size, linewidth=line_width_vec[i],
                     color=color_vec[i])

# Fig: 5_indoor_classify, 6_outdoor_classify
def plot_classification_acc():
    dfs = pd.read_excel("classification accuracy.xlsx", sheet_name="Indoor")
    legend_vec = dfs['Methods'][0::5]
    sub_num = 7
    data = dfs.values[:, 2:sub_num+2].astype(np.float)
    # Fig: 5_indoor_classify
    plot_acc_mean_std_subject(data, '5_indoor_classify', sub_num, legend_vec)

    # Fig: 6_outdoor_classify
    dfs = pd.read_excel("classification accuracy.xlsx", sheet_name="Outdoor")
    data = dfs.values[:, 2:sub_num+2].astype(np.float)
    plot_acc_mean_std_subject(data, '6_outdoor_classify', sub_num, legend_vec)


# Fig: 7_time_delay_analysis
def plot_classification_acc_with_delays():
    legend_vec = ['CNN', 'CNN + Voting', 'CNN + RNN + Voting',
                  'CNN + LSTM + Voting', 'CNN + GRU + Voting', 'Ours']
    fig = plt.figure(figsize=(9, 5))
    plt.rcParams.update({'font.size': 15})
    sheet_name_vec = ['Indoor', 'Outdoor']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.xticks([0, 5, 10])
        plt.ylim(0.90, 1)
        plt.yticks(np.arange(0.85, 1.01, 0.05), ['85', '90', '95', '100'])
        plt.ylabel(sheet_name_vec[i] + ' classification accuracy (%)')
        plt.xlabel('Delayed frames')
        dfs = pd.read_excel("classification accuracy for different delays.xlsx",
                            sheet_name=sheet_name_vec[i])
        acc_mean_mat = dfs.values[1:12,1:1+len(legend_vec)].astype(np.float)
        acc_std_mat = dfs.values[1:12,1+len(legend_vec):1+2 * len(legend_vec)].astype(np.float)
        plot_error_line(np.arange(1,12, dtype = np.int), acc_mean_mat, acc_std_mat)
    fig.tight_layout()
    fig.legend(legend_vec, loc='lower center', ncol=3, bbox_to_anchor=(0.49, 0.95))
    plt.savefig('images/7_time_delay_analysis.pdf', bbox_inches='tight', pad_inches=0.15)
    plt.show()


# Fig: 8_indoor_decision, 9_outdoor_decision
def plot_decisions():
    legend_vec = ['CNN', 'CNN + Voting', 'CNN + RNN + Voting',
                  'CNN + LSTM + Voting', 'CNN + GRU + Voting', 'Ours', 'Actual']
    sheet_name_vec = ['Indoor', 'Outdoor']
    fig_name_vec = ['images/8_indoor_decision.pdf', 'images/9_outdoor_decision.pdf']
    marker_vec = ['o', '+', 'v', 'x', 'd', '*', '']
    line_vec = ['', '', '', '', '', '', '-']
    linewidth_vec = [1, 1, 1, 1, 1, 1, 5]
    for i in range(2):
        fig = plt.figure(figsize=(9, 5))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 15})
        plt.ylim(0.9, 5.1)
        plt.yticks(np.arange(1, 5.1, 1), ['LG', 'US', 'DS', 'UR', 'DR'])
        plt.ylabel('Modes')
        plt.xlabel('Time (s)')
        dfs = pd.read_excel("decisions.xlsx",
                            sheet_name=sheet_name_vec[i])
        time_vec = dfs['Time (s)'][:]
        y_mat = dfs.values[:, 1:].astype(np.float)
        plot_line(time_vec, y_mat, marker_vec = marker_vec, line_vec = line_vec,
                  line_width_vec= linewidth_vec)
        plt.legend(legend_vec, loc='lower center', ncol=3, bbox_to_anchor=(0.49, 1.0))
        plt.savefig(fig_name_vec[i], bbox_inches='tight')
    plt.show()


# Fig: 10_probability_analysis
def plot_probability_and_decisions():
    sheet_name_vec = ['Indoor']
    fig = plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 15})
    dfs = pd.read_excel("probabilities and decisions.xlsx", sheet_name='Indoor')
    time_vec = dfs['Time (s)'][1:]
    y_label_vec = ['Probability\n(CNN)', 'Probability\n(Ours)', 'Modes']
    legend_vec = ['LG', 'US', 'DS', 'UR', 'DR']
    x_label_vec = ['Time (s)\n(a)', 'Time (s)\n(b)', 'Time (s)\n(c)']
    for i in range(2):
        plt.subplot(3, 1, i + 1)
        plt.tight_layout()
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.5))
        plt.ylabel(y_label_vec[i])
        plt.xlabel(x_label_vec[i])
        y_mat = dfs.values[1:, 1 + i * 5:1 + (i + 1) * 5].astype(np.float)
        plot_line(time_vec, y_mat, only_plot_error=False, marker_size=5,
                  is_last_black=False)
        plt.legend(legend_vec, loc='lower center', ncol=5, bbox_to_anchor=(0.49, 1.0))

    i = 2
    plt.subplot(3, 1, i + 1)
    plt.ylim(0.9, 5.1)
    plt.yticks(np.arange(1, 5.1, 1), ['LG', 'US', 'DS', 'UR', 'DR'])
    plt.ylabel(y_label_vec[i])
    plt.xlabel(x_label_vec[i])
    y_mat = dfs.values[1:, 1 + i * 5:1 + i * 5 + 3].astype(np.float)
    marker_vec = ['o', '*', '']
    line_vec = ['', '', '-']
    line_width_vec = [1, 1, 5]
    plot_line(time_vec, y_mat, marker_vec=marker_vec,
              line_vec=line_vec, line_width_vec=line_width_vec)
    legend_vec = ['CNN', 'Ours', 'Actual']
    plt.legend(legend_vec, loc='lower center', ncol=3, bbox_to_anchor=(0.49, 1.0))
    fig.tight_layout()
    plt.savefig('images/10_probability_analysis.pdf', bbox_inches='tight')
    plt.show()


# Check if data is normal distributed and calculate p values
def calc_p_matrix():
    sheet_name_vec = ['Indoor', 'Outdoor']
    sub_num = 7
    is_normal_list = [True, True]
    p_matrix = np.ones((2, 6, 6))
    for i in range(2):
        dfs = pd.read_excel("classification accuracy.xlsx", sheet_name=sheet_name_vec[i])
        data = dfs.values[:, 2:sub_num+2].astype(np.float)
        data_list = []
        for m in range(6):
            data_m = data[5*m:5*(m+1),:].reshape(-1)
            if is_normal_list[i]:
                statistic, critical_values, significance_level = stats.anderson(data_m)
                if statistic > critical_values[2]:
                    is_normal_list[i] = False
            data_list.append(data_m)
        for r in  range(6):
            for c in  range(6):
                if is_normal_list[i]:
                    _, p_matrix[i, r, c] = stats.ttest_ind(data_list[r], data_list[c])
                else:
                    _, p_matrix[i, r, c] = stats.ranksums(data_list[r], data_list[c])
    return p_matrix, is_normal_list

# Check if data is normal distributed and calculate p values
# p_matrix, is_normal_list = calc_p_matrix()

# # Fig: 5_indoor_classify, Fig: 6_outdoor_classify
# plot_classification_acc()

# Fig: 7_time_delay_analysis
plot_classification_acc_with_delays()

# Fig: 8_indoor_decision, 9_outdoor_decision
# plot_decisions()

# Fig: 10_probability_analysis
# plot_probability_and_decisions()
