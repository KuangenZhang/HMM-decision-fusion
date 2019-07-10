%%
clear all;
close all;
clc;
addpath('D:\4-code\3rd_parties\Matlab\subaxis\');
dataPath = '../3-save_data/';
load([dataPath,'classification_DifferentMethods.mat']);
addpath('D:/4-code/project/VisionRobot/2-matlab/visionProcess/');
%% segment the signals to the sequence; split data to training, validation, and test set
[x,y] = cell_2_seq_array(scoreVecCell, correctLabelsCell, [1,9]);
len_y = size(y,2);
idx_vec = randperm(len_y);
split_idx = floor(len_y/2);
x_train = x(:,:,1:split_idx);
y_train = y(:,1:split_idx);
x_val = x(:,:,split_idx+1:end);
y_val = y(:,split_idx+1:end);

[x_test,y_test] = cell_2_seq_array(scoreVecCell, correctLabelsCell, [2:8,10:16]);
%% save h5 file
data_cell = {x_train, x_val, x_test};
label_cell = {y_train, y_val, y_test};
h5FileNameVec = {[dataPath,'training_set.h5'], [dataPath,'/validataion_set.h5'],...
    [dataPath,'test_set.h5']};
for set_num = 1:3
    data = data_cell{set_num};
    label = label_cell{set_num};
    createH5File(h5FileNameVec{set_num},size(data));
    writeH5File(h5FileNameVec{set_num},data,label); 
end
fprintf('Finished!');
%%
function createH5File(fileName, data_size)
h5create(fileName,'/data',data_size,'Datatype','single');
h5create(fileName,'/label',[1 data_size(end)],'Datatype','uint8');
end

function writeH5File(fileName,data,label)
data_size = size(data)
h5write(fileName,'/data',data,[1 1 1], data_size);
h5write(fileName,'/label',label,[1 1], [1 data_size(end)]);
h5disp(fileName)
end

function [x_sequence, y_sequence] = seg_signals(x_signals, y_signals, win_length)
    y_sequence = y_signals(win_length:end)';
    x_size = size(x_signals);
    len_y_sequence = size(y_sequence,2);
    x_sequence = zeros(x_size(end), win_length,len_y_sequence);
    for i = 1:len_y_sequence
        x_sequence(:,:, i) = x_signals(i:(i+win_length-1),:)';
    end
end
function [x,y] = cell_2_seq_array(scoreVecCell, correctLabelsCell, col_vec)
x = [];
y = [];
win_length = 15;
for c = col_vec
    for r = 1:5
        [x_sequence, y_sequence] = seg_signals(scoreVecCell{r,c},...
            correctLabelsCell{r,c}, win_length);
        x = cat(3,x, x_sequence);
        y = cat(2,y, y_sequence);
    end
end
end