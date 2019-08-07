%%
clear all;
close all;
clc;
addpath('D:\4-code\3rd_parties\Matlab\subaxis\');
dataPath = 'D:/4-code/project/VisionRobot/0-dataSet/2_Validset_EFRS_processed/';
load([dataPath,'classification_DifferentMethods.mat']);
addpath('D:/4-code/project/VisionRobot/2-matlab/visionProcess/');
%%
priorMat = [0.8,0.05,0.05,0.05,0.05;
    0.15,0.8,0.01,0.03,0.01;
    0.15,0.01,0.8,0.01,0.03;
    0.15,0.03,0.01,0.8,0.01;
    0.15,0.01,0.03,0.01,0.8];
backWinLenVec = 1:2:9;
midWinLenVec = 1:2:11;
meanAccuracyPredictMat  = zeros(max(backWinLenVec),11);
meanAccuracyMidFiltMat  = zeros(max(backWinLenVec),11);
meanAccuracyHmmMat   = zeros(max(backWinLenVec),11);
accuracyPredictMat = zeros(5,8);
accuracyBackFiltMat = zeros(5,8);
accuracyMidFiltMat = zeros(5,8);
accuracyBackMidFiltMat = zeros(5,8);
accuracyPriorFiltMat = zeros(5,8);
%%
for backWinLen = backWinLenVec
    backWinLen
    for midWinLen = midWinLenVec
        [accCNN,accVoting,accHmm]...
            = DataProcess.calcDecisionAccuracy(priorMat,scoreVecCell,predictLabelsCell,...
            correctLabelsCell,backWinLen,midWinLen);
        meanAccuracyPredictMat(backWinLen, midWinLen) = accCNN(1);
        meanAccuracyMidFiltMat(backWinLen, midWinLen) = accVoting(1);
        meanAccuracyHmmMat(backWinLen, midWinLen) = accHmm(1);
    end
end
%% 5,6-ClassificationResults
backWinLen = 5;
midWinLen = 1;
accCNNMat = cell(1,2);
accVotingMat = cell(1,2);
accHmmMat = cell(1,2);
for place = 1:2
    subIdx = 8*(place-1) + [1:8];
    [accCNNMat{place}, accVotingMat{place},accHmmMat{place}]...
        = DataProcess.calcDecisionAccuracyMat(priorMat,...
        scoreVecCell(:,subIdx),predictLabelsCell(:,subIdx),...
        correctLabelsCell(:,subIdx),backWinLen,midWinLen);
end
fprintf('Finished.\n')
%% 5,6-ClassificationResults
accuracyCell = cell(8,2);
for place = 1:2
    for subNum = 1:8
        accuracyCell{subNum,place} = [accCNNMat{place}(:,subNum),...
            accVotingMat{place}(:,subNum),accHmmMat{place}(:,subNum)];
    end
end
save('saveData/classificationAccuracy.mat','accCNNMat','accVotingMat','accHmmMat');
%% 5,6- Check if results are normal distributed
for i = 1:2
    is_noraml_cnn(i) = adtest(accCNNMat{i}(:))
    is_noraml_hmm(i) = adtest(accHmmMat{i}(:))
    is_noraml_voting(i) = adtest(accVotingMat{i}(:))
end
% ttest (indoor) and Wilcoxon rank sum test (outdoor)
acc_cell = {accCNNMat,accVotingMat, accHmmMat};
for r = 1:3
    for c = 1:3
    [~, p_indoor(r,c)] = ttest(acc_cell{r}{1}(:), acc_cell{c}{1}(:));
    end
end
for r = 1:3
    for c = 1:3
    p_outdoor(r,c) = ranksum(acc_cell{r}{2}(:), acc_cell{c}{2}(:));
    end
end
%% 5,6-ClassificationResults
close all;
figurePath = 'D:\2-science\OneDrive - alumni.ubc.ca\1-PhDWork\21-papers\4-Decision fusion\2-images\';
% idx = 1; % indoor
% figName = [figurePath,'5_indoor_classify'];
idx = 2; % outdoor
figName = [figurePath,'6_outdoor_classify'];
fig = figure('DefaultAxesFontSize',15, 'Units', 'centimeters','Position',[0,0,16,12]);
subaxis(1,1,1, 'Spacing', 0.00, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
legendVec = {'CNN','CNN + Voting','Ours'};
PlotClass.plotErrorBarForCellOrigin(accuracyCell(:,idx),...
    'Subject number','Indoor classification accuracy (%)',legendVec);
[mean_vec, std_vec, groups, p_val_vec, accuracy_mat] = DataProcess.census_cell(accuracyCell(:,idx))
set(gca,'xtick',2:5:37,'XLim',[0,40],'xticklabel',...
    {'S1';'S2';'S3';'S4';'S5';'A1';'A2';'A3'},'ytick',0.75:0.05:1,...
    'YTickLabel',{'75','80','85','90','95','100'},'YLim',[0.75,1]...
    ,'FontName', 'Times New Roman','FontSize',15,'LooseInset',get(gca,'TightInset'));
FileIO.printFig(fig,figName);
save(figName, 'mean_vec', 'std_vec', 'groups', 'p_val_vec', 'accuracy_mat')
%% 7-Time delay analysis / HmmMidWindowComparison
backWinLen = 5;
midWinLenVec = 1:2:21;
accMean = zeros(length(midWinLenVec),3,2);
accStd = zeros(length(midWinLenVec),3,2);
time_delay_vec = (midWinLenVec+1)/2;
for midWinLen = midWinLenVec
    for place = 1:2
        subIdx = 8*(place-1) + [2:8];
        [accCNN,accVoting,accHmm]...
            = DataProcess.calcDecisionAccuracy(priorMat,...
            scoreVecCell(:,subIdx),predictLabelsCell(:,subIdx),...
            correctLabelsCell(:,subIdx),backWinLen,midWinLen);
        accMean((midWinLen+1)/2,:,place) = [accCNN(1),accVoting(1),accHmm(1)];
        accStd((midWinLen+1)/2,:,place) = [accCNN(2),accVoting(2),accHmm(2)];
    end
end
%%
acc_mean_cell = cell(1,2);
acc_std_cell = cell(1,2);
for i = 1:2
    acc_mean_cell{i} = accMean(:,:,i);
    acc_std_cell{i} = accStd(:,:,i);
end
fprintf('Finished.\n')
%% 7-calculate gradient
voting_gradient = (accMean(end,2,:) - accMean(1,2,:))/10
our_gradient = (accMean(end,3,:) - accMean(1,3,:))/10
%% 7-HmmMidWindowComparison
close all;
figurePath = 'D:\2-science\OneDrive - alumni.ubc.ca\1-PhDWork\21-papers\4-Decision fusion\2-images\';
figName = [figurePath,'7_time_delayed_analysis'];
fig = figure('DefaultAxesFontSize',10,'Units', ...
    'centimeters','Position',[0,0,12,12]);
lenData = length(midWinLenVec);
types = [{'CNN'};{'CNN + Voting'};{'Ours'}];
lineStyle = [{'-.'};{'--'};{':'}];
markerStyle = [{'+'};{'o'};{'s'}];
ylabelVec = [{'Indoor environmental classification accuracy (%)'}; 
    {'Outdoor environmental classification accuracy (%)'}];
colorVec = colormap('lines');
colorVec = colorVec([1,3,4],:);
for place = 1:2
    subaxis(1,2,place, 'Spacing', 0.05, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
    hold on;
    for i = 1:3
        e = errorbar(time_delay_vec,accMean(:,i,place),accStd(:,i,place),...
            'LineWidth',2,'Marker',markerStyle{i},'LineStyle',lineStyle{i},...
            'Color',colorVec(i,:));
        e.Bar.LineStyle = e.Line.LineStyle;
        e.Line.LineStyle = 'solid';
    end
    hold off;
    xlabel('Delayed frames');
    xlim([0,max(time_delay_vec)+1])
    ylabel(ylabelVec{place});
    if place == 2
        legend(types,'Location','northoutside','Orientation','horizontal');
    end
    set(gca,'FontName', 'Times New Roman','FontSize',9,...
        'ytick',0.85:0.05:1, 'YTickLabel',{'85','90','95','100'},...
        'YLim',[0.85,1]);
    set(gca,'LooseInset',get(gca,'TightInset'));
end
FileIO.printFig(fig,figName);
%% 8,9-DecisionModesComparison
close all;
backWinLen = 5;
midWinLen = 3;
exe_num = 1;
%Indoor
% % fileName = [dataPath,'Amputee_1/Amputee_1_indoor_set_actual (1)'];
% fileName = [dataPath,'Amputee_1/Amputee_1_indoor_set (', num2str(exe_num),')'];
% % figName = [figurePath,'8_indoor_decision'];
% subNum = 6;

%Outdoor
% fileName = [dataPath,'Amputee_2/Amputee_2_outdoor_set_actual (1)'];
fileName = [dataPath,'Amputee_2/Amputee_2_outdoor_set (', num2str(exe_num),')'];
% figName = [figurePath,'9_outdoor_decision'];1
subNum = 15;
predictLabels = predictLabelsCell{exe_num,subNum};
correctLabels = correctLabelsCell{exe_num,subNum};
midFiltLabels = DataProcess.votingFilt1(predictLabels,midWinLen+2,'mid');
scoreVec = scoreVecCell{exe_num,subNum};
backHmmStates = DataProcess.calcOptimalStates(scoreVec,priorMat,backWinLen);
backHmmStates = DataProcess.votingFilt1(backHmmStates,midWinLen,'mid');
actualModes = DataProcess.readFileToLabels(fileName);
timesVec = timeVecCell{exe_num,subNum};
legendVec = {'Actual','CNN','CNN + Voting','Ours'};
yLabelVec = {'LG' 'US' 'DS','UR','DR'};
fig = figure('DefaultAxesFontSize',13.5,'Units', ...
    'centimeters','Position',[0,0,16,8]);
subaxis(1,1,1, 'Spacing', 0.00, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
plotDecisionModes(predictLabels,midFiltLabels,...
    backHmmStates,actualModes,timesVec,legendVec,yLabelVec);
% FileIO.printFig(fig,figName);
%% 10-ProbabilityVariation
backWinLen = 5;
midWinLen = 3;
figurePath = 'D:\2-science\OneDrive - alumni.ubc.ca\1-PhDWork\21-papers\4-Decision fusion\2-images\';
figName = [figurePath,'10_probability_analysis'];
fig = figure('DefaultAxesFontSize',10,'Units', ...
    'centimeters','Position',[0,0,12,20]);
% for k = 6
for k = 6
%     for exeNum = 3
    for exeNum = 3
        scoreVec = scoreVecCell{exeNum,k};
        predictLabels = predictLabelsCell{exeNum,k};
        correctLabels = correctLabelsCell{exeNum,k};
        timeVec = timeVecCell{exeNum,k};
        lineStyle= [{'-'};{'.'};{'--'};{':'};{'-.'}];
        cmap = lines(6);
        subaxis(3,1,1, 'Spacing', 0.05, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
        PlotClass.plotMat(timeVec, scoreVec, [{'LG'};{'US'};{'DS'};{'UR'};{'DR'}],...
            lineStyle,cmap);
        ylabel({'Probability';'(CNN)'});
        [hmmStates,score_vec_hmm] = DataProcess.calcOptimalStates(scoreVec,priorMat,backWinLen);
        hmmStates = DataProcess.votingFilt1(hmmStates,midWinLen,'mid');
        subaxis(3,1,2, 'Spacing', 0.05, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
        PlotClass.plotMat(timeVec, score_vec_hmm, [{'LG'};{'US'};{'DS'};{'UR'};{'DR'}],...
            lineStyle,cmap);
        ylabel({'Probability';'(Ours)'});
        subaxis(3,1,3, 'Spacing', 0.05, 'PaddingLeft', 0.05, 'PaddingBottom', 0.05);
        legendVec = [{'Actual','CNN','Ours'}];
        lineStyle= [{'-'};{'o'};{'--'}];
        cmap=cmap([4,1,4],:);
        cmap(1,:) = [0,0,0];
        PlotClass.plotClassificationResults(timeVec,...
            [{correctLabels};{predictLabels};{hmmStates}],legendVec,...
            lineStyle,cmap);
        xlabel('Time (s)')
    end
end
FileIO.printFig(fig,figName);
%% plot
legendVec = [{'Correct labels','CNN results','CNN+HMM results'}];
lineStyle= [{'-'};{'o'};{'--'}];
PlotClass.plotClassificationResults(timeVec,...
    [{correctLabels};{predictLabels};{hmmStates}],legendVec,lineStyle);
%% 10-HmmWithDifferentWIndowComparison
meanAccuracyCell = cell(1,3);
types = [{'CNN'};{'CNN + Voting'};{'CNN + HMM + Voting'}];
lineStyle = [{':'};{'--'};{'-'}];
colorVec = colormap('lines');
colorVec = colorVec([1,3,4],:);
meanAccuracyCell{1} = meanAccuracyPredictMat;
meanAccuracyCell{2} = meanAccuracyMidFiltMat;
meanAccuracyCell{3} = meanAccuracyHmmMat;
% save('meanAccuracy.mat','meanAccuracyCell','types');
fprintf('Finished.\n')
typeNumVec = 1:3;
% figure('DefaultAxesFontSize',9,'Units', ...
%     'centimeters','Position',[0,0,12,12]);
hold on;
[rows,cols] = meshgrid(midWinLenVec,backWinLenVec);
for i = typeNumVec
    surf(rows,cols,meanAccuracyCell{i}(backWinLenVec,midWinLenVec),...
        'FaceColor',colorVec(i,:),'FaceAlpha', 0.5,'LineStyle',lineStyle{i});
end
hold off;
view([0.6,1,0.2])
xlabel('Voting window length','Rotation',5);
ylabel('HMM window length','Rotation',-15);
zlabel('Accuracy')
legend(types(typeNumVec),'Location','north','Orientation','horizontal');
set(gca,'FontName', 'Times New Roman','FontSize',9,'LooseInset',get(gca,'TightInset'));
%% 12-all labels generation;
hmmBackWinLen = 5;
hmmMidWinLen = 3;
votingWinLen = 9;
[midVoteLabelsCell(:,1:8), backHmmStatesCell(:,1:8)] = ...
    DataProcess.calcLabelsCell(priorMat,scoreVecCell(:,1:8),predictLabelsCell(:,1:8),...
    votingWinLen,hmmBackWinLen,hmmMidWinLen);
votingWinLen = 19;
[midVoteLabelsCell(:,9:16), backHmmStatesCell(:,9:16)] = ...
    DataProcess.calcLabelsCell(priorMat,scoreVecCell(:,9:16),predictLabelsCell(:,9:16),...
    votingWinLen,hmmBackWinLen,hmmMidWinLen);
fprintf('Finished.\n')
%%
fileName=[dataPath,...
    datestr(now,'mm-dd-yyyy-HH-MM-SS'),'_PredictScoresWithTime.mat'];
save(fileName,'scoreVecCell','predictLabelsCell','midVoteLabelsCell',...
    'backHmmStatesCell','correctLabelsCell','timeVecCell','timeJumpCell','priorMat');
fprintf('Finished.\n');
%% 12-TimeDelayComparision
close all;
tic
isPlot = 0;
%1:16
%1:5
timeJumpCell = cell(size(timeVecCell));
timeJumpCellOrigin = cell(size(timeVecCell));
%%
for subNum = 1:16
    subNum
    if subNum < 8
        votingWinLen = 9;
    else
        votingWinLen = 19;
    end
    for exeNum = 1:5
        exeNum
        timesVec = timeVecCell{exeNum,subNum};
        delayLen = round(1/ mean(diff(timesVec)));
        correctLabels = correctLabelsCell{exeNum,subNum};
        correctLabels(delayLen+1:end) = correctLabels(1:end-delayLen);
        midVoteLabels = midVoteLabelsCell{exeNum,subNum};
        midVoteLabels((votingWinLen+1)/2:end) = midVoteLabels(1:end-(votingWinLen-1)/2);
        backHmmStates = backHmmStatesCell{exeNum,subNum};
        backHmmStates((hmmMidWinLen+3)/2:end) = backHmmStates(1:end-(hmmMidWinLen+1)/2);
        labelsMat = [correctLabels,midVoteLabels,backHmmStates];
        predictLabels = predictLabelsCell{exeNum,subNum};
        for i = 2:3
            labelsMat(:,i) = DataProcess.votingFilt1(labelsMat(:,i),31,'mid');
        end
        if isPlot
            legendVec = {'Actual modes','CNN','CNN + Voting','CNN + HMM + Voting'};
            yLabelVec = {'LG' 'US' 'DS','UR','DR'};
            PlotClass.plotDecisionModes(predictLabels,labelsMat(:,2),...
                labelsMat(:,3),labelsMat(:,1),timesVec,legendVec,yLabelVec);
        end
        timeJumpMat = [];
        for i =1:3
            diffLabels = [0;diff(labelsMat(:,i))];
            timeJump = timesVec(diffLabels~=0);
            timeJump(timeJump < 2) = [];
            diffTimeJump = [timeJump(1);diff(timeJump)];
            timeJump(diffTimeJump < 1.3) = []
            timeJumpCellOrigin{exeNum,subNum} = [timeJumpCellOrigin{exeNum,...
                subNum};timeJump];
            if i == 1
                timeJumpMat = [timeJumpMat,timeJump];
            else
                timeJumpMat = [timeJumpMat,timeJump(1:length(timeJumpMat(:,1)))];
            end
        end
        timeJumpCell{exeNum,subNum} = timeJumpMat;
    end
end
toc
fprintf('Finished.\n')
%% 12-TimeDelayComparision
timeDelayCell = cell(1,2);
for place = 1:2
    timeDelay = [];
    for subNum = (place-1)*8+(1:8)
        for exeNum = 1:5
            timeJumpMat = timeJumpCell{exeNum, subNum};
            timeDelayTemp = [timeJumpMat(:,1)-timeJumpMat(:,2:3)];
            if(max(max(abs(timeDelayTemp)))) > 5
                subNum
                exeNum
            end
            timeDelay = [timeDelay;timeDelayTemp];
        end
    end
    timeDelayCell{place} = timeDelay;
end
%% 12-TimeDelayComparision: plot
legendVec = {'CNN + Voting','CNN + HMM + Voting'};
% PlotClass.plotBoxForCell(timeDelayCell,'','Average lead time (%)',legendVec)
PlotClass.plotErrorBarForCell(timeDelayCell,'','Average lead time (%)',legendVec)
set(gca,'xtick',1.5:4:5.5,'XLim',[0,7],'xticklabel',...
    {'Indoor';'Outdoor'}...
    ,'FontName', 'Times New Roman','FontSize',9,'LooseInset',get(gca,'TightInset'));



function plotDecisionModes(predictedModes,filteredModes,...
    decisionModes,actualModes,timeStamps,legendVec,yLabelVec)
hold on;
fPlot = [];
cmap = lines(8);
acPlot = plot(timeStamps,actualModes,'k','LineWidth',4);
prePlot = plot(timeStamps,predictedModes,'o','LineWidth',2,'MarkerSize',4,'Color', cmap(1, :));
filPlot = plot(timeStamps,filteredModes,'b','LineWidth',2,'Color', cmap(3, :));
dePlot = plot(timeStamps,decisionModes,'g--','LineWidth',2,'Color', cmap(4, :));
% for video
hold off;
legend([fPlot,acPlot,prePlot,filPlot,dePlot],legendVec,...
    'Location','northoutside','Orientation','horizontal');
xlabel('Time (s)');
ylabel('Modes');
yticks(1:1:5);
ylim([0.7 5]);
xlim([0,max(timeStamps) + 2])
yticklabels(yLabelVec);
set(gca,'FontName', 'Arial',...
    'FontSize',12,'LooseInset',get(gca,'TightInset'));
end