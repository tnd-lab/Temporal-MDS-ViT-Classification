% plot range-angle heatmap
function [axh] = plot_rangeAng(Xcube,rng_grid,agl_grid, save_path)

if nargin < 4
   save_path = [];
end

Nr = size(Xcube,1);   %%%length of Chirp(num of rangeffts)
Ne = size(Xcube,2);   %%%number of angleffts
Nd = size(Xcube,3);   %%%length of chirp loop

% polar coordinates
for i = 1:size(agl_grid,2)
    yvalue(i,:) = (sin(agl_grid(i)*pi/180 )).*rng_grid;
end
for i=1:size(agl_grid,2)
    xvalue(i,:) = (cos(agl_grid(i)*pi/180)).*rng_grid;
end

%% plot 2D(range-angle)

Xpow = abs(Xcube);
Xpow = squeeze(sum(Xpow,3)/size(Xpow,3));

% noisefloor = db2pow(-15);
Xsnr=Xpow;
% Xsnr = pow2db(Xpow/noisefloor);

if ~isempty(save_path)
    fig = figure('visible','off');
    % figure()
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(agl_grid,rng_grid,Xsnr);
    view(0,90)
    % axis([-90 90 0 25]);
    axis([-60 60 2 25]);
    grid off
    shading interp
    xlabel('Angle of arrive(degrees)')
    ylabel('Range(meters)')
    colorbar
    caxis([0,0.2])
    title('Range-Angle heatmap')
    
    % Save the figure if a path is provided
    if nargin > 1 && ~isempty(save_path)
        [folder, ~, ~] = fileparts(save_path);
        if ~isempty(folder) && ~isfolder(folder)
            mkdir(folder); % Create the directory if it doesn't exist
        end
        saveas(fig, save_path); % Save the figure in the specified format
    end
else
    ig = figure('visible','on');
    % figure()
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(agl_grid,rng_grid,Xsnr);
    view(0,90)
    % axis([-90 90 0 25]);
    axis([-60 60 2 25]);
    grid off
    shading interp
    xlabel('Angle of arrive(degrees)')
    ylabel('Range(meters)')
    colorbar
    caxis([0,0.2])
    title('Range-Angle heatmap')
end

end