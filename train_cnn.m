folder = 'train_folder\';

imds = imageDatastore(folder, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

rng(123);
figure;
perm = randperm(200,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds)

img = readimage(imds,1);
size(img)

p = 0.6;
[imdsTrain,imdsValidation] = splitEachLabel(imds,p,'randomize');



%%
layers = [
    imageInputLayer([42 42 1])
    
    convolution2dLayer(3,4,'Padding','same','Name','conv4')
    batchNormalizationLayer('Name','batchnorm4')
    reluLayer('Name','relu4')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,8,'Padding','same','Name','conv8')
    batchNormalizationLayer('Name','batchnorm8')
    reluLayer('Name','relu8')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same','Name','conv16')
    batchNormalizationLayer('Name','batchnorm16')
    reluLayer('Name','relu16')
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same','Name','conv32')
    batchNormalizationLayer('Name','batchnorm32')
    reluLayer('Name','relu32')
    
     maxPooling2dLayer(2,'Stride',2)
     
    convolution2dLayer(3,64,'Padding','same','Name','conv64')
    batchNormalizationLayer('Name','batchnorm64')
    reluLayer('Name','relu64')
      
    
    fullyConnectedLayer(10,'Name','fully_connected')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification_output')];


%%
options = trainingOptions('adam', ...
    'InitialLearnRate',0.005, ...
    'MaxEpochs',60, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',4, ...
    'Verbose',false, ...
    'Plots','training-progress');


%%
[net,data] = trainNetwork(imdsTrain,layers,options);
%%
figure;

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
confusionchart(YPred,YValidation);