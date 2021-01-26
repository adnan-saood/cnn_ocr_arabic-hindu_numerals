load('percent95.mat');
net.Layers(end).Classes

I = deepDreamImage(net,'fully_connected',8,'NumIterations',50, ...
    'PyramidLevels',4);

%%
figure
I = imtile(I);
imshow(I)