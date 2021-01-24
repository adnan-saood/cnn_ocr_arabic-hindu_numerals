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
    
    convolution2dLayer(3,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    ر 