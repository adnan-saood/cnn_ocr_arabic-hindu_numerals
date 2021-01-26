function [I] = create_activations(layer_name,number)
load('percent95.mat','net');
rng(123);
im = imread(['samples\' num2str(number) '.bmp']);
if strcmp(layer_name,'fully_connected') | strcmp(layer_name,'softmax') | strcmp(layer_name, 'classification_output')
    v = 1;
else
    v = 0;
end
if v == 0
    act1 = activations(net,im,layer_name);
    
    sz = size(act1);
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    
    I = imtile(mat2gray(act1),'GridSize',[ceil(sqrt(sz(3))) ceil(sqrt(sz(3)))]);
else
    act1 = activations(net,im,layer_name);
    
    sz = size(act1);
    
    act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
    
    act2 = [act1(:,:,10),...
            act1(:,:,5),...
            act1(:,:,9),...
            act1(:,:,8),...
            act1(:,:,3),...
            act1(:,:,2),...
            act1(:,:,7),...
            act1(:,:,6),...
            act1(:,:,1),...
            act1(:,:,4)];
    
    I = mat2gray(act2);
end
end

