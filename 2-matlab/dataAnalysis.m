%% test a fit and classfy algorithm
clear all;
close all;
clc;
dataPath = 'D:/4-code/project/VisionRobot/0-dataSet/2_Validset_EFRS_processed/';
addpath(dataPath);
addpath('D:/4-code/project/VisionRobot/2-matlab/visionProcess/');
subjectNameVec = [{'Subject_'};{'Amputee_'}];
placeNameVec = [{'_indoor_set'};{'_outdoor_set'}]; 
placeName = placeNameVec{1};
load('D:/4-code/project/VisionRobot/2-matlab/visionProcess/nets/gregnet_5Classes_simu.mat');
winLen = 7;
isIndoor = true;
accuracyFiltVec = zeros(5,8);%classification accuracy
%% Analysis scores 
for k = 1:8
    fprintf([num2str(k),'\n']);
    if k > 5
        subjectName = subjectNameVec{2};
        subNum = k - 5;
    else
        subjectName = subjectNameVec{1};
        subNum = k;
    end
    medianFeatures = struct('groundParas',[],'upStairParas',[],'downStairParas',[],...
                'upRampParas',[],'downRampParas',[]);
    for exeNum = 1:5
        fileName = [subjectName,num2str(subNum),'/',subjectName,...
            num2str(subNum),placeName,' (',num2str(exeNum),')'];
        [actualModes,correctLabels] = DataProcess.initLabelsFromFile([dataPath,fileName]);
        load([dataPath,fileName,'.mat']);
        [predictLabels,decisionLabels,accuracy,filteredLabels,timeStamps] = ...
            ImgAlgo.classifyImgs(imgAndAngleData,gregnet_5Classes_simu,correctLabels,actualModes,winLen);
        accuracyFiltVec(exeNum,k) = accuracy(2);
    end
    mean(accuracyFiltVec(:,k))
end
mean(mean(accuracyFiltVec))
fprintf('Finished.\n');
%% plot accuracy results
% figure('DefaultAxesFontSize',8,'Units', 'centimeters','Position',[0,0,18,18]);
PlotClass.plotStdErrorBar(accuracyFiltVec);
%%
export_fig([figurePath,'Outdoor.png'],'-transparent');
export_fig([figurePath,'Outdoor.fig']);
%% plot
figure;
PlotClass.plotDecisionModes(predictLabels,filteredLabels,...
    decisionLabels,actualModes,[1:length(imgAndAngleData)]);
%% test a fit and classfy algorithm
addpath(dataPath);
addpath('D:/4-code/project/VisionRobot/2-matlab/visionProcess/');
subjectNameVec = [{'Subject_'};{'Amputee_'}];
placeNameVec = [{'_indoor_set'};{'_outdoor_set'}];
load('D:/4-code/project/VisionRobot/visionProcess/nets/gregnet_5Classes_Final.mat');
isIndoor = true;
accuracyOldMat = zeros(5,8);%classification accuracy
accuracyNewMat = zeros(5,8);%classification accuracy
%% output binary and rgb image;
for k = 6:8
    fprintf([num2str(k),'\n']);
    if k > 5
        subjectName = subjectNameVec{2};
        subNum = k - 5;
    else
        subjectName = subjectNameVec{1};
        subNum = k;
    end
    for exeNum = 1:5
        fileName = [subjectName,num2str(subNum),'/',subjectName,...
            num2str(subNum),placeName,' (',num2str(exeNum),')'];
        load([dataPath,fileName,'.mat']);
        ImgAlgo.saveImgRGBAndBinary(imgAndAngleData,fileName,dataPath);
    end
end
fprintf('All finished.\n');
%% load data
for k = 6
    fprintf([num2str(k),'\n']);
    if k > 5
        subjectName = subjectNameVec{2};
        subNum = k - 5;
    else
        subjectName = subjectNameVec{1};
        subNum = k;
    end
    for exeNum = 1
        fileName = [subjectName,num2str(subNum),'/',subjectName,...
            num2str(subNum),placeName,' (',num2str(exeNum),')'];
        load([dataPath,fileName,'.mat']);
    end
end
%% output 3D point cloud, 2D scatter, and binary images
fileFloder = ['D:\2-science\OneDrive - alumni.ubc.ca\'...,
    '1-PhDWork\21-papers\4-HMM for sequential environmental classification\1-images\1-figure\2-FeaturesExtraction\']
% i = 1, 350, 700, 1000, 1400;
i = 50;
% pointsXYZ = vecDownRamp{1};
% pointsXYZ = vecDownStairs{1};
% pointsXYZ = vecUpRamp{1};
% pointsXYZ = vecUpStairs{1};
% pointsXYZ = vecGround{1};
pointsXYZ = imgAndAngleData(i).pointsXYZ;
figure('DefaultAxesFontSize',9,'Units', ...
    'centimeters','Position',[0,0,8,24]);
subplot(3,1,1);
pcshow(pointsXYZ);
ylim([-1,0])
% pointsXYZ = pointsXYZ(abs(pointsXYZ(:,2)) < 0.01,:);
% pointsXZ = pointsXYZ(:,[1,3]);
pointsXZ = imgAndAngleData(i).pointsXZ;
% pointsXZ(pointsXZ(:,1) < 0.15,:) = [];
subplot(3,1,2);
scatter(pointsXZ(:,1),pointsXZ(:,2),'.');
ylim([-1,0])
binaryImg = ImgAlgo.setOccupancyMap(pointsXZ,1);
subplot(3,1,3);
imshow(binaryImg);
saveas(gcf,[fileFloder,num2str(i)],'epsc')
saveas(gcf,[fileFloder,num2str(i)],'fig')
%         imwrite(binaryImg,[fileFolder,'/',num2str(i),'.png']);
%% extract time and save with classification results
timeVecCell = cell(5,16);
for k = 1:16
    fprintf([num2str(k),'\n']);
    subjectName = subjectNameVec{1};
    if k > 8
        placeName = placeNameVec{2};
        if k > 13
            subjectName = subjectNameVec{2};
            subNum = k - 13;
        else
            subNum = k - 8;
        end
    else
        placeName = placeNameVec{1};
        if k > 5
            subjectName = subjectNameVec{2};
            subNum = k - 5;
        else
            subNum = k;
        end
    end
    for exeNum = 1:5
        fileName = [subjectName,num2str(subNum),'/',subjectName,...
            num2str(subNum),placeName,' (',num2str(exeNum),')'];
        [actualModes,correctLabels] = ...
            DataProcess.initLabelsFromFile([dataPath,fileName]);
        loadData = load([dataPath,fileName,'.mat']);
        imgAndAngleData = loadData.imgAndAngleData;
        timeVec = zeros(length(imgAndAngleData),1);
        parfor idx = 1:length(imgAndAngleData)
            timeVec(idx) = imgAndAngleData(idx).timeStamp;
        end
        timeVec = timeVec - timeVec(1);
        timeVecCell{exeNum,k} = timeVec;
    end
end
%%
fileName=[dataPath,...
    datestr(now,'mm-dd-yyyy-HH-MM-SS'),'_PredictScoresWithTime.mat'];
save(fileName,'scoreVecCell','predictLabelsCell',...
    'correctLabelsCell','timeVecCell','priorMat');
fprintf('Finished.\n');