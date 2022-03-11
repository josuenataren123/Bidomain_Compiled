function noiseValue = Perlin_Noise_Jakes(X, Y, g, IMG_W, IMG_H)

centerX = floor(IMG_W / 2);
centerY = floor(IMG_H / 2);


x = centerX + floor(X);
y = centerY + floor(Y);
% 
% x = X;
% y = Y;

if x < 0
    x = -x;
%     elseif x >= IMG_W
%         x = IMG_W-1;
end
if y < 0
    y = -y;
%     elseif y >= IMG_H
%         y = IMG_H-1;
end

if x == 1 || x == 0
    x = 2;
end
if y == 1 || y == 0
    y = 2;
end

X = x;
Y = y;

%[gX,gY] = gradient(WhiteNoise);


%All points at the same distance: since we are looking at the point in the
%middle of a cell or square formed by the nodes, the distance from the
%middle to the nodes is always .5 movement in x and .y movement in y (unit grid assumption)
%Therefore, all the alpha functions will have the same value.

%Top right corner
alphaTR_x = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
alphaTR_y = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
rTR = [.5, .5];
gTR = [ g(X+1,Y+1, 1); g(X+1,Y+1, 2)];
TRPoint_Contribution = alphaTR_x*alphaTR_y*( rTR*gTR );


%Top left corner
alphaTL_x = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
alphaTL_y = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
rTL = [-.5, .5];
gTL = [ g(X-1,Y+1, 1); g(X-1,Y+1, 2)];
TLPoint_Contribution = alphaTL_x*alphaTL_y*( rTL*gTL );


%Bottom right corner
alphaBR_x = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
alphaBR_y = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
rBR = [.5, -.5];
gBR = [ g(X+1,Y-1, 1); g(X+1,Y-1, 2)];
BRPoint_Contribution = alphaBR_x*alphaBR_y*( rBR*gBR );


%Bototm Left point
alphaBL_x = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
alphaBL_y = 6*(1 - .5)^5 - 15*(1 - .5)^4 + 10*(1 - .5)^3;
rBL = [-.5, -.5];
gBL = [ g(X-1,Y-1, 1); g(X-1,Y-1, 2)];
BLPoint_Contribution = alphaBL_x*alphaBL_y*( rBL*gBL );


%Total value
noiseValue = TRPoint_Contribution + TLPoint_Contribution + BRPoint_Contribution + BLPoint_Contribution;

end