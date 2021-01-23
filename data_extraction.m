raw2 = imread('arabicocr.bmp');
raw = im2bw(raw2,0.5);
imshow(raw);
%%
offset_x = 4+5;
offset_y = 9+5;
w = 51 - 4/10;
l = 52.3 - 2/10;
W = 42;
L = 42;

%%
hold on;
for i = 0:9
    
    for j = 0:19
        y = round(i*w + offset_x: (i*w + W + offset_x-1));
        x = round(j*l + offset_y: (j*l + L + offset_y-1));
        temp(:,:,(i)*20 + (j+1)) = raw(x ,y);
      
        scatter(y , x , 'k.');
    end
end

%%
for o = 1:size(temp,3)
    imwrite(temp(:,:,o),[ 'train\tr' num2str(o) '.bmp']);
    
end