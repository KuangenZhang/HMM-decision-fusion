%%
clear all;
close all;
clc;
dataPath = '../3-save_data/';
load([dataPath,'classification_DifferentMethods.mat']);
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
%% save subject_test_set
is_seg = true;
save_test_set(scoreVecCell, correctLabelsCell, [2:8,10:16], dataPath, is_seg);
is_seg = false;
save_test_set(scoreVecCell, correctLabelsCell, [2:8,10:16], dataPath, is_seg);
fprintf('Finished!');
%%
function createH5File(file_name, data_size, data_name, label_name)
if nargin < 4
    data_name = '/data';
    label_name = '/label';
end
if isempty(h5read(file_name,data_name))
    h5create(file_name,data_name,data_size,'Datatype','single');
end
if isempty(h5read(file_name,label_name))
    h5create(file_name,label_name,[1 data_size(end)],'Datatype','uint8');
end
end

function writeH5File(file_name,data,label, data_name, label_name)
if nargin < 5
    data_name = '/data';
    label_name = '/label';
end
data_size = size(data);
h5write(file_name,data_name,data,ones(1,length(data_size)), data_size);
h5write(file_name,label_name,label,[1 1], [1 data_size(end)]);
h5disp(file_name)
end

function [x_sequence, y_sequence] = seg_signals(x_signals, y_signals, win_length)
    y_sequence = y_signals(win_length-1:end-1)';
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

function save_test_set(scoreVecCell, correctLabelsCell, col_vec, dataPath, is_seg)
win_length = 15;
if is_seg
    file_name = [dataPath,'subject_test_set.h5'];
else
    file_name = [dataPath,'score_vec.h5'];
end

for c = col_vec
    for r = 1:5
        if is_seg
            [data, label] = seg_signals(scoreVecCell{r,c},...
            correctLabelsCell{r,c}, win_length);
        else
            data = scoreVecCell{r,c}';
            label = correctLabelsCell{r,c}';
        end
        data_name = ['/data_r_',num2str(r),'_c_',num2str(c)];
        label_name = ['/label_r_',num2str(r),'_c_',num2str(c)];
        createH5File(file_name,size(data), data_name, label_name);
        writeH5File(file_name,data,label, data_name, label_name); 
    end
end
end