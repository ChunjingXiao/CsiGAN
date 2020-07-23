%% xiao
% % this is for semi-supervised learning: extract unlabelled data from leaven-one user
% this selects "selectCateogry" categories and igore other categories. each category has 10 samples
% for example, if trainset: useer 2, 3, and 4;;  test: user 1;
% unlabelCategory: select "unlabelCategory" categories as unlabel data; unlabel size is unlabelCategory*unlabelNumEnd
% unlabel: 1--unlabelNumEnd    per category in first "unlabelCategory" categories from user 1
% testset: testNumStart--10 per category in all selectCateogry categories from user 1

    clear
    dataDir = 'data';
    load([dataDir '/csi_tensorUnlabel_all.mat']);
    load([dataDir '/labelUnlabel_all.mat']);
    
    load([dataDir '/cycleCsi_tensorTrain_all.mat']);
    load([dataDir '/cycleLabelTrain_all.mat']);
    
    selectCateogry = 50;  % select selectCateogry for train, test and unlabel
    unlabelCategory = 10; % select unlabelCategory categories as unlabel data; unlabel size unlabelCategory*unlabelNumEnd
    unlabelCategoryCycle = 5;
    
    % select unlabelCategory categories from csi_tensorUnlabel_all for unlabel
    csi_tensorUnlabel = [];
%     allCategoryRand = randperm(selectCateogry);
%     unlabelSelect = allCategoryRand(1:unlabelCategory);
%     testSelect =  allCategoryRand(unlabelCategory+1:end);
%     for i= unlabelSelect
    for i=1:unlabelCategory
        arrayNum = find(labelUnlabel_all == i);
        csi_tensorUnlabel = cat(4,csi_tensorUnlabel, csi_tensorUnlabel_all(:,:,:,arrayNum(1:end)));
    end
    
    %--------cycle--------begin------ % select unlabelCategory categories from cycleCsi_tensorTrain_all for cycle train  
    cycleCsi_tensorUnlabel = [];
    for i=1:unlabelCategoryCycle
        arrayNum = find(labelUnlabel_all == i);
        cycleCsi_tensorUnlabel = cat(4,cycleCsi_tensorUnlabel, csi_tensorUnlabel_all(:,:,:,arrayNum(1:end)));
    end
    
    cycleLabelTrain_all = cycleLabelTrain_all - selectCateogry;
    cycleCsi_tensorTrain = [];
    for i=1:unlabelCategoryCycle
        arrayNum = find(cycleLabelTrain_all == i);
        cycleCsi_tensorTrain = cat(4,cycleCsi_tensorTrain, cycleCsi_tensorTrain_all(:,:,:,arrayNum(1:end)));
    end
    %--------cycle--------end-------
    

   
    fprintf('size(csi_tensorUnlabel)       : %s\n', num2str(size(csi_tensorUnlabel)))
    fprintf('size(cycleCsi_tensorTrain)    : %s\n', num2str(size(cycleCsi_tensorTrain)))    
    fprintf('size(cycleCsi_tensorUnlabel)  : %s\n', num2str(size(cycleCsi_tensorUnlabel))) 

    
    csi_tensorUnlabel = randomRankOne(csi_tensorUnlabel);
    cycleCsi_tensorTrain = randomRankOne(cycleCsi_tensorTrain);
    cycleCsi_tensorUnlabel = randomRankOne(cycleCsi_tensorUnlabel);
    
    
    save([dataDir '/csi_tensorUnlabel'],'csi_tensorUnlabel')
    
    save([dataDir  '/cycleCsi_tensorTrain'],'cycleCsi_tensorTrain')         %--------used to train CycleGAN------
    
    save([dataDir  '/cycleCsi_tensorUnlabel'],'cycleCsi_tensorUnlabel')         %--------used to train CycleGAN------

    disp('done!') 


    
    
function [csiData,labelData] = randomRankTwo(csi_tensorTrain,labelTrain)
    sampleSizeTrain = size(labelTrain);
    rrTrain = randperm(sampleSizeTrain(1));
    csiData = csi_tensorTrain(:,:,:,rrTrain);
    labelData = labelTrain(rrTrain);
end
function csiData = randomRankOne(csi_tensorUnlabel)
    sampleSizeUnlabel = size(csi_tensorUnlabel);
    rrUnlabel = randperm(sampleSizeUnlabel(4));
    csiData = csi_tensorUnlabel(:,:,:,rrUnlabel);
end

