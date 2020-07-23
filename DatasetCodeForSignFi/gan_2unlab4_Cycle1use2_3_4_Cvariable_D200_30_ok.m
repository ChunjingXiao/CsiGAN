%% xiao
% % this is for semi-supervised learning: extract unlabelled data from leaven-one user
% this selects "selectCateogry" categories and igore other categories. each category has 10 samples
% for example, if trainset: useer 2, 3, and 4;;  test: user 1;
% unlabelCategory: select "unlabelCategory" categories as unlabel data; unlabel size is unlabelCategory*unlabelNumEnd
% unlabel: 1--unlabelNumEnd    per category in first "unlabelCategory" categories from user 1
% testset: testNumStart--10 per category in all selectCateogry categories from user 1

    clear
    load('dataset_lab_150.mat');
    label1 = [label(1:1250);   label(1251:1500)-128];
    label2 = [label(1501:2750);label(2751:3000)-128];
    label3 = [label(3001:4250);label(4251:4500)-128];
    label4 = [label(4501:5750);label(5751:6000)-128];  
    %label5 = [label(6001:7250);label(7251:7500)-128]; 
    if(1)
        csi1 = abs(csi1);
        csi2 = abs(csi2);
        csi3 = abs(csi3);
        csi4 = abs(csi4); 
        %csi5 = abs(csi5);
    else
        csi1 = [abs(csi1), angle(csi1)];
        csi2 = [abs(csi2), angle(csi2)];
        csi3 = [abs(csi3), angle(csi3)];
        csi4 = [abs(csi4), angle(csi4)];
        %csi5 = [abs(csi5), angle(csi5)];
%         csi1 = csi1(:,1:2:end,:,:);
%         csi2 = csi2(:,1:2:end,:,:);
%         csi3 = csi3(:,1:2:end,:,:);
%         csi4 = csi4(:,1:2:end,:,:);
%         csi5 = csi5(:,1:2:end,:,:);
    end
    % -------------------------------------------only change this----------begin------------------    
    trainCsi1 = csi2;     %----trainset-----
    trainlabel1 = label2;
    trainCsi2 = csi3;
    trainlabel2 = label3;
    trainCsi3 = [];       % trainCsi3 =[];   % trainCsi3= csi4;
    trainlabel3 = [];      % trainlabel3=[];  % trainlabel3= label4;
    trainCsi4 = [];        % trainCsi4=[];    % trainCsi4= csi5;
    trainlabel4 = [];      % trainlabel4=[];  % trainlabel4= label5;
    
    leaveCsi   = csi1;    %-----leave this----
    leaveLabel = label1;
    
    selectCateogry = 50;  % select selectCateogry for train, test and unlabel
    unlabelCategory = 10; % select unlabelCategory categories as unlabel data; unlabel size unlabelCategory*unlabelNumEnd
    unlabelNumEnd = 5;    % select 1--unlabelNumEnd per category for unlabel data
    testNumStart = 6;     % select testNumStart--10 per category for test set

    % setting: trainCsi3 = []; trainCsi4 = []; selectCateogry = 50; unlabelCategory = 10; unlabelNumEnd =5; testNumStart = 6; 
    % means: trainset: user 2 and 3;; test: testNumStart--10 samples per category of user 1;;
    %        unlabel: 1--unlabelNumEnd per category of user 1(1 per category from left 50% of user 1)

    % PS: there are no disjoint among train, test and unlabel
    % -------------------------------------------only change this----------end--------------------
    cycleCsi_user2 = trainCsi2;       %--------cycle--------
    cycleLabelUser2 = trainlabel2;    %--------cycle--------
    
    csiTrain = cat(4,trainCsi1,trainCsi2,trainCsi3,trainCsi4);
    labelTrain = [trainlabel1;trainlabel2;trainlabel3;trainlabel4];
    %csiTrain = cat(4,csiUse2,csiUse3);
    %labelTrain = [labelUse2;labelUse3];
    
    [leaveCsi,leaveLabel] = randomRankTwo(leaveCsi,leaveLabel);  % rank randomly ---- Leave

    % select selectCateogry categories for train and test (ignore other categories)
    trainAdd = [];         
    testAdd = [];
    trainLabelAdd = [];
    testLabelAdd = [];
    for i=1:selectCateogry
        %disp(i)
        arrayTrain = find(labelTrain == i);   
        trainAdd = cat(4,trainAdd, csiTrain(:,:,:,arrayTrain(1:end)));        
        trainLabelAdd = [trainLabelAdd; labelTrain(arrayTrain(1:end))];
        arrayTest = find(leaveLabel == i); 
        testAdd  = cat(4,testAdd, leaveCsi(:,:,:,arrayTest(1:end)));
        testLabelAdd  = [testLabelAdd ; leaveLabel(arrayTest(1:end))];
    end
    csiTrain = trainAdd;
    labelTrain = trainLabelAdd;
    csiTest = testAdd;
    labelTest2 = testLabelAdd;
    
    csi_tensorTrain = csiTrain;               % training data
    csi_tensorTest2 = csiTest;                % test data
    
    
   % for each category in selectCateogry, select 1--unlabelNumEnd for unlabel, select testNumStart--10(end) for test 

    csi_tensorTest = [];
    labelTest = [];
    
    csi_tensorUnlabel_all = []; 
    labelUnlabel_all = [];
    cycleCsi_tensorTrain_all = [];        %--------cycle--------begin------
    cycleLabelTrain_all = [];            %--------cycle--------end-------

    for i=1:selectCateogry
        %disp(i)
        arrayNum = find(labelTest2 == i);
        % for each category in selectCateogry, select 1--unlabelNumEnd to csi_tensorUnlabel 
        csi_tensorUnlabel_all = cat(4,csi_tensorUnlabel_all, csi_tensorTest2(:,:,:,arrayNum(1:unlabelNumEnd)));
        labelUnlabel_all = [labelUnlabel_all;labelTest2(arrayNum(1:unlabelNumEnd))];
        % for each category in selectCateogry, select testNumStart--10(end) to csi_tensorTest 
        csi_tensorTest  = cat(4,csi_tensorTest, csi_tensorTest2(:,:,:,arrayNum(testNumStart:end)));
        labelTest = [labelTest;labelTest2(arrayNum(testNumStart:end))];
        
        %--------cycle--------begin------ % select 1--unlabelNumEnd from one train user for cycle train
        arrayNum = find(cycleLabelUser2 == i);
        cycleCsi_tensorTrain_all = cat(4,cycleCsi_tensorTrain_all, cycleCsi_user2(:,:,:,arrayNum(1:unlabelNumEnd)));
        cycleLabelTrain_all = [cycleLabelTrain_all;cycleLabelUser2(arrayNum(1:unlabelNumEnd))];
        %--------cycle--------end-------
    end

    
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
    cycleCsi_tensorTrain = [];
    for i=1:unlabelCategory
        arrayNum = find(cycleLabelTrain_all == i);
        cycleCsi_tensorTrain = cat(4,cycleCsi_tensorTrain, cycleCsi_tensorTrain_all(:,:,:,arrayNum(1:end)));
    end
    cycleLabelTrain_all = cycleLabelTrain_all + selectCateogry;
    %--------cycle--------end-------
    

    fprintf('size(csi_tensorTrain)         : %s\n', num2str(size(csi_tensorTrain)))
    fprintf('size(labelTrain)              : %s\n', num2str(size(labelTrain)))
    fprintf('size(csi_tensorTest)          : %s\n', num2str(size(csi_tensorTest)))    
    fprintf('size(labelTest)               : %s\n', num2str(size(labelTest)))
    
    fprintf('size(csi_tensorUnlabel)       : %s\n', num2str(size(csi_tensorUnlabel)))   
    fprintf('size(csi_tensorUnlabel_all)   : %s\n', num2str(size(csi_tensorUnlabel_all)))
    fprintf('size(labelUnlabel_all)        : %s\n', num2str(size(labelUnlabel_all)))
        
    fprintf('size(cycleCsi_tensorTrain)    : %s\n', num2str(size(cycleCsi_tensorTrain)))    
    fprintf('size(cycleCsi_tensorTrain_all): %s\n', num2str(size(cycleCsi_tensorTrain_all)))
    fprintf('size(cycleLabelTrain_all)     : %s\n', num2str(size(cycleLabelTrain_all)))
    
    
    % rank randomly
    [csi_tensorTrain,labelTrain] = randomRankTwo(csi_tensorTrain,labelTrain);  
    [csi_tensorTest,labelTest] = randomRankTwo(csi_tensorTest,labelTest); 
    csi_tensorUnlabel = randomRankOne(csi_tensorUnlabel);
    
    cycleCsi_tensorUnlabel = csi_tensorUnlabel;
    
    save('data/csi_tensorTrain','csi_tensorTrain') 
    save('data/labelTrain','labelTrain')
    save('data/csi_tensorTest','csi_tensorTest')
    save('data/labelTest','labelTest')
    save('data/csi_tensorUnlabel','csi_tensorUnlabel')
    
    save('data/csi_tensorUnlabel_all','csi_tensorUnlabel_all')
    save('data/labelUnlabel_all','labelUnlabel_all')        
    
    save('data/cycleCsi_tensorTrain','cycleCsi_tensorTrain')         %--------used to train CycleGAN------
    save('data/cycleCsi_tensorTrain_all','cycleCsi_tensorTrain_all') %--------used to generate fake labeled samples------
    save('data/cycleLabelTrain_all','cycleLabelTrain_all')           %--------used to generate fake labeled samples------
    save('data/cycleCsi_tensorUnlabel','cycleCsi_tensorUnlabel') 
    disp('done!') 
    n_epoch = 100;
    [net_info,perf]=signfi_cnn_user1to4_leave1_Cvariable...
        (csi_tensorTrain,csi_tensorTest,labelTrain,labelTest,selectCateogry,n_epoch);  

    
    
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

