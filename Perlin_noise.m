%Jakes et. al. 2019 Perlin Noise


%PARAMETERS TO USE
%phi_fiber can change based on whatever desired for fiber orientation
%                     lb    gamma    R    phi_fiber    d     ld     f     L
%   Interstitial      .24    .96    1.89     34       .32   4.67   .30   .31
%   Compact           .96    .59    2.47     -9       .44   2.03    0     -
%   Diffuse           .07    .44    2.17     11       .49   1.22    0     -
%   Patchy            .32    .78    2.50     68       .42   2.10   .38   .31

%Density (according to table, but just suggested, also can vary based on
%desired density):
%Interstitial: 9.6%
%Compact: 47.2%
%Diffuse: 22.0%
%Patchy: 26.9%

%Base Noise field
lb = .32;             %[.01, 2]   Base size of fibrotic obstacles (mm)
lb = lb/10;           %now in cm
R = 2.50;                %[.5, 50]   Anisotropy of base obstacles
gamma = .78;           %[0, .99]   Roughness of base noise field

%Density Variation Noise field
ld = 2.10;               %[1, 8]   Size of density variation features (mm)
ld = ld/10;           %now in cm

%Fiber selection Noise field
L = .31;               %[.3, 2]   Base separation distance of fibres
L = L/10;
phi_fiber_orientation = 68;                             %[-pi/2, pi/2]   Fibre orientation in degrees
phi_fiber_orientation = -phi_fiber_orientation*pi/180;          %now in radians

%Others
f = .38;                     %[0, .4]   Extent of pattern coincidence with fibres
d = .42;                    %[0, .5]   Magnitude of local density variation

Seed = 1;
numEl = 512;
width = numEl;

fibrosisDensity_desired = 26.9;% in percentage: 20 = 20%, 30 = 30%, 40 = 40%, etc. not decimals

%Fixed: Fiber selection Noise field
s = 15;   %Sharpening factor
phi_phase_modulation = 10*pi;   %Strength of phase modulation
Beta_dissimilarity = .25;   %Dissimilarity between fibres
Beta_wiggliness = .25;   %Fibre ?wiggliness?

Beta_dissimilarity = Beta_dissimilarity*5;
Beta_wiggliness = Beta_wiggliness*60;

Nf = 4; %number of octaves of Perlin noise


IMG_W = width;    
IMG_H = width;

A_R = [1./sqrt(R), 0; 0, sqrt(R)];
R_phi = [cos(phi_fiber_orientation), sin(phi_fiber_orientation); -sin(phi_fiber_orientation), cos(phi_fiber_orientation)];

%%
%CREATING GEOMETRY

%cm
maxX = 1;
minX = -1;
maxY = 1;
minY = -1;
dx = (maxX - minX)/numEl;
dy = (maxY - minY)/numEl;

[X,Y] = meshgrid(linspace(minX, maxX, numEl),linspace(minY, maxY, numEl));
figure
plot(X,Y,"or")

% for i=1:numEl
%     for j=1:numEl
%         plot(X,Y,"ob")
%         hold on
%         plot(X(i,j), Y(i,j), "ro")
%         pause(.1)
%     end
% end

%%
%IMAGING WHITE NOISE 

CMAP = [0 0 0 ; hsv(255)];%HSV colormap
img = ones(IMG_H,IMG_W);
% initiate figure
hFig = figure('Menu','none', 'Pointer','crosshair', 'DoubleBuffer','on');


% initiate axis and colormap
hAx = axes('Color','k', 'XLim',[0 IMG_W]+0.5, 'YLim',[0 IMG_H]+0.5, ...
    'Units','normalized', 'Position',[0 0 1 1]);
colormap(hAx, CMAP);

% initiate image
hImg = image('XData',1:IMG_W, 'YData',1:IMG_H, ...
    'CData',img, 'CDataMapping','direct');

TrialVal = 5;

WhiteNoise = zeros(IMG_W*TrialVal,IMG_H*TrialVal);
   
% draw the whitenoise
for x = 1:IMG_W*TrialVal
  for y = 1:IMG_H*TrialVal
        clr = (1-rand()*2);
        WhiteNoise(x,y) = clr;
  end
end

randomVectorGrid = zeros(IMG_W*TrialVal,IMG_H*TrialVal, 2);
   
% draw the whitenoise
for x = 1:IMG_W*TrialVal
  for y = 1:IMG_H*TrialVal
        v= 2*rand(2,1)-1;
        n=v/sqrt(v(1)^2+v(2)^2);
        randomVectorGrid(x,y,:) = n;
  end
end

for x = 1:IMG_W
  for y = 1:IMG_H
        img(y,x) = (WhiteNoise(x,y)+1)*125;
  end
end
set(hImg, 'CData',img)
drawnow  
        
        
%%
%CREATING PERLIN NOISE FIELDS FOR THE THREE DIFFERENT NOISE FIELDS
%JUST DO MAPPING FROM GRID TO NOISE FIELD
% draw the perlin noise

%three noisefields:
% - Nb_field: Fibrotic Obstacles
% - Nd_field: Density Variation
% - F_field: Fiber Selection



Nb_perlinMatrix = zeros(IMG_H, IMG_W);
Nd_perlinMatrix = zeros(IMG_H, IMG_W);
F_perlinMatrix = zeros(IMG_H, IMG_W);

Nb_field_img = ones(IMG_H,IMG_W);
Nd_field_img = ones(IMG_H,IMG_W);
F_field_img = ones(IMG_H,IMG_W);

I_perlinMatrix = zeros(IMG_H, IMG_W);
I_final = ones(IMG_H,IMG_W);

[gX,gY] = gradient(WhiteNoise);
% gX = rescale(gX); 
% gY = rescale(gY); 



for x = 1:IMG_W
    for y = 1:IMG_H
        %N_b BASE PERLIN NOISEFIELD
        % offsetV1 = -.5 + (.5+.5) .* rand(1,1);
        %perlin = perlinNoise2d(x, y, 0.1, 4, IMG_W, IMG_H, WhiteNoise);
        xV_Nb = (1./lb).*([X(x,y), Y(x,y)]*A_R);
%         xV_Nb = (1./lb).*([x, y]*A_R);
        perlinNb = perlinNoise2d(xV_Nb(1), xV_Nb(2), gamma, Nf, IMG_W, IMG_H, WhiteNoise, randomVectorGrid);
        %perlin = perlinNoise2d(50*X(x,y)+2, 10*Y(x,y)+2, 0.1, 4, IMG_W, IMG_H, WhiteNoise);
        Nb_perlinMatrix(x,y) = perlinNb;
        cRGB_Nb = floor(1+ (perlinNb - 0) / (1 - 0) * (256-1)); 
        %cRGB_Nb = floor(5.3857 + (perlinNb - (-4.3230)) / (5.3857 - (-4.3230)) * (256-1)); 
        Nb_field_img(x,y) = cRGB_Nb;
        
        
        %N_d DENSITY VARIATION PERLIN NOISEFIELD
        xV_Nd = (1./ld).*([X(x,y), Y(x,y)]);
        perlinNd = perlinNoise2d(xV_Nd(1), xV_Nd(2), .5, Nf, IMG_W, IMG_H, WhiteNoise, randomVectorGrid);
        Nd_perlinMatrix(x,y) = perlinNd;
        cRGB_Nd = floor(1+ (perlinNd - 0) / (1 - 0) * (256-1)); 
        %cRGB_Nd = floor(1.7932 + (perlinNd - (-1.3009)) / (1.7932 - -1.3009) * (256-1));
        Nd_field_img(x,y) = cRGB_Nd;
        
        
        %F FIBER SELECTION NOISEFIELD
        
        %With rotation included
        xV_F = (([X(x,y), Y(x,y)] * (R_phi)) * [(Beta_wiggliness), 0; 0, (Beta_dissimilarity/L)] );
        
        %Without rotation included
%         xV_F = (([X(x,y), Y(x,y)]) * [(Beta_wiggliness), 0; 0, (Beta_dissimilarity./L)] );
        
        Z_field = (phi_phase_modulation).*perlinNoise2d(xV_F(1), xV_F(2), .5, Nf, IMG_W, IMG_H, WhiteNoise, randomVectorGrid);
        perlinF = (.5 + .5*cos( (2*pi/L)*xV_F(2) + Z_field ) ).^s;
        F_perlinMatrix(x,y) = perlinF;
        %cRGB_F = floor(1+ (perlinF - 0) / (1 - 0) * (256-1));
        cRGB_F = floor(1+ (perlinF - 0) / (1 - 0) * (256-1));
        F_field_img(x,y) = cRGB_F;
        
        
        %FINAL IMAGE (DETERMINE IF ROTATION OR NO ROTATION)
        %if rotation, rotate first all the points in X,Y
        I_perlinMatrix(x,y) = ((perlinNb)*((1 - f) + (f*perlinF))) + d*perlinNd;
        cRGB_I = floor(1+ (I_perlinMatrix(x,y) - 0) / (1 - 0) * (256-1)); 
        I_final(x,y) = cRGB_I;
        
        
    end
end
% 
% Nb_perlinMatrix = rescale(Nb_perlinMatrix); 
% Nd_perlinMatrix = rescale(Nd_perlinMatrix); 
% F_perlinMatrix = rescale(F_perlinMatrix); 
% 
% for x = 1:IMG_W
%     for y = 1:IMG_H
%         cRGB_Nb = floor(1+ (Nb_perlinMatrix(x,y) - 0) / (1 - 0) * (256-1)); 
%         %cRGB_Nb = floor(5.3857 + (perlinNb - (-4.3230)) / (5.3857 - (-4.3230)) * (256-1)); 
%         Nb_field_img(x,y) = cRGB_Nb;
%         
%         
%         
%         cRGB_Nd = floor(1+ (Nd_perlinMatrix(x,y) - 0) / (1 - 0) * (256-1)); 
%         %cRGB_Nd = floor(1.7932 + (perlinNd - (-1.3009)) / (1.7932 - -1.3009) * (256-1));
%         Nd_field_img(x,y) = cRGB_Nd;
%         
%         
%         
%         %cRGB_F = floor(1+ (perlinF - 0) / (1 - 0) * (256-1));
%         cRGB_F = floor(1+ (F_perlinMatrix(x,y) - 0) / (1 - 0) * (256-1));
%         F_field_img(x,y) = cRGB_F;
%         
%         
%         %FINAL IMAGE (DETERMINE IF ROTATION OR NO ROTATION)
%         %if rotation, rotate first all the points in X,Y
%         I_perlinMatrix(x,y) = ((Nb_perlinMatrix(x,y))*((1 - f) + (f*F_perlinMatrix(x,y)))) + d*Nd_perlinMatrix(x,y);
%         cRGB_I = floor(1+ (I_perlinMatrix(x,y) - 0) / (1 - 0) * (256-1)); 
%         I_final(x,y) = cRGB_I;
%         
%         
%     end
% end




%DETERMINING THRESHOLD FOR FIBROSIS IMAGING
meanV = mean(mean(I_final));
stdV = std(std(I_final));
maxV = max(max(I_final));
minV = min(min(I_final));

colorFibrosis = 18;% RED COLOR
colorTissue = 46;% YELLOW COLOR

%fibroticThreshold = 100;
fibroticThreshold = prctile(reshape(I_final,1,[]),(100-fibrosisDensity_desired));
fibCount = 0;
totCount = 0;

%histogram(I_final, 20)

for x = 1:IMG_W
    for y = 1:IMG_H
        if I_final(x,y) > fibroticThreshold
            I_final(x,y) = colorFibrosis;
            fibCount = fibCount + 1;
            totCount = totCount + 1;
        else
            I_final(x,y) = colorTissue;
            totCount = totCount + 1;
        end
    end
end

fibrosis_density = fibCount/totCount;


% figure

% initiate figure for Nb
hFig2 = figure('Pointer','crosshair', 'DoubleBuffer','on');
% initiate axis and colormap
hAx1 = axes('Color','k', 'XLim',[0 IMG_W]+0.5, 'YLim',[0 IMG_H]+0.5, ...
    'Units','normalized', 'Position',[0 0 1 1]);
colormap(hAx1, CMAP);
% initiate image
hImg1 = image('XData',1:IMG_W, 'YData',1:IMG_H, ...
    'CData',Nb_field_img, 'CDataMapping','direct');
set(hImg1, 'CData',Nb_field_img)
drawnow  


% initiate figure for Nd
hFig3 = figure('Pointer','crosshair', 'DoubleBuffer','on');
% initiate axis and colormap
hAx2 = axes('Color','k', 'XLim',[0 IMG_W]+0.5, 'YLim',[0 IMG_H]+0.5, ...
    'Units','normalized', 'Position',[0 0 1 1]);
colormap(hAx2, CMAP);
% initiate image
hImg2 = image('XData',1:IMG_W, 'YData',1:IMG_H, ...
    'CData',Nd_field_img, 'CDataMapping','direct');
set(hImg2, 'CData',Nd_field_img)
drawnow  


% initiate figure for F
hFig4 = figure('Pointer','crosshair', 'DoubleBuffer','on');
% initiate axis and colormap
hAx3 = axes('Color','k', 'XLim',[0 IMG_W]+0.5, 'YLim',[0 IMG_H]+0.5, ...
    'Units','normalized', 'Position',[0 0 1 1]);
colormap(hAx3, CMAP);
% initiate image
hImg3 = image('XData',1:IMG_W, 'YData',1:IMG_H, ...
    'CData',F_field_img, 'CDataMapping','direct');
set(hImg3, 'CData',F_field_img)
drawnow  


% initiate figure for final Image
hFig5 = figure('Pointer','crosshair', 'DoubleBuffer','on');
% initiate axis and colormap
hAx4 = axes('Color','k', 'XLim',[0 IMG_W]+0.5, 'YLim',[0 IMG_H]+0.5, ...
    'Units','normalized', 'Position',[0 0 1 1]);
colormap(hAx4, CMAP);
% initiate image
hImg4 = image('XData',1:IMG_W, 'YData',1:IMG_H, ...
    'CData',I_final, 'CDataMapping','direct');
set(hImg4, 'CData',I_final)
drawnow  



%plotting in grid itself
% fibroticInd = find(I_final == colorFibrosis);
% tissueInd = find(I_final == colorTissue);
% 
% figure
% plot(X(tissueInd),Y(tissueInd),"oy")
% hold on
% plot(X(fibroticInd),Y(fibroticInd),"or")
% grid on

% these indices are the ones that will be used to set the specific
%conductivities differently.
       
        
%%
%FUNCTIONS TO BE USED


 function r = cosineInterpolate(a, b, x)
    ft = x*pi;
    f = (1-cos(ft))*0.5;
    r = (a*(1-f)+b*f);
end

function re = smoothedNoise2d(x,y, IMG_W, IMG_H, WhiteNoise)
    
    centerX = floor(IMG_W / 2);
    centerY = floor(IMG_H / 2);
    
    x = centerX + x;
    y = centerY + y;
    
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
    
    
%     x
%     y


    
    corners = (WhiteNoise(x-1,y-1) + WhiteNoise(x+1,y-1) + WhiteNoise(x-1,y+1) + WhiteNoise(x+1,y+1))/16;
    sides = (WhiteNoise(x-1,y) + WhiteNoise(x+1,y) + WhiteNoise(x,y+1) + WhiteNoise(x,y-1))/8;
    center = WhiteNoise(x,y)/4;
    re = (corners + sides + center);
    
end


function rb = interpolatedNoise2d(x,y, IMG_W, IMG_H, WhiteNoise)
    integerX = floor(x);
    integerY = floor(y);

    fractionalX = x-integerX;
    fractionalY = y-integerY;

    v1 = smoothedNoise2d(integerX,integerY, IMG_W, IMG_H, WhiteNoise);
    v2 = smoothedNoise2d(integerX+1,integerY, IMG_W, IMG_H, WhiteNoise);
    v3 = smoothedNoise2d(integerX, integerY+1, IMG_W, IMG_H, WhiteNoise);
    v4 = smoothedNoise2d(integerX+1,integerY+1, IMG_W, IMG_H, WhiteNoise);

    i1 = cosineInterpolate(v1,v2,fractionalX);
    i2 = cosineInterpolate(v3,v4, fractionalX);

    rb = cosineInterpolate(i1,i2,fractionalY);
end




function res = perlinNoise2d(x,y, gamma, Nf, IMG_W, IMG_H, WhiteNoise, g)
    P_total = 0;
    
    for i = 1:Nf
        frequency = 2^(i-1); 
        amplitude = gamma^(i-1);
        offsetV = -.5 + (.5+.5).*rand(1,1);
%         offsetVector(i) = offsetV;
%         offsetV = 0;
        P_total = P_total + interpolatedNoise2d(x*frequency + offsetV,y*frequency + offsetV, IMG_W, IMG_H, WhiteNoise)*amplitude;
%         P_total = P_total + Perlin_Noise_Jakes(x*frequency + offsetV, y*frequency + offsetV, g, IMG_W, IMG_H)*amplitude;
        
    end

    res = 1/2 + ((sqrt(2)*(1 - gamma))/(2*(1 - gamma^Nf))) * P_total;
end

