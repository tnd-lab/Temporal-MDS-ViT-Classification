% plot range-angle heatmap
function [axh] = plot_rangeDop(Dopdata_sum,rng_grid,vel_grid, save_path)

% plot 2D(range-Doppler)
if nargin < 4
   save_path = [];
end

if ~isempty(save_path)
    fig = figure('visible','off')
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(vel_grid,rng_grid,Dopdata_sum);
    view(0,90)
    axis([-8 8 2 25]);
    grid off
    shading interp
    xlabel('Doppler Velocity (m/s)')
    ylabel('Range(meters)')
    colorbar
    caxis([0,3e04])
    title('Range-Doppler heatmap')
    
    % Save the figure if a path is provided
    if nargin > 1 && ~isempty(save_path)
        [folder, ~, ~] = fileparts(save_path);
        if ~isempty(folder) && ~isfolder(folder)
            mkdir(folder); % Create the directory if it doesn't exist
        end
        saveas(fig, save_path); % Save the figure in the specified format
    end
else
    fig = figure('visible','on')
    set(gcf,'Position',[10,10,530,420])
    [axh] = surf(vel_grid,rng_grid,Dopdata_sum);
    view(0,90)
    axis([-8 8 2 25]);
    grid off
    shading interp
    xlabel('Doppler Velocity (m/s)')
    ylabel('Range(meters)')
    colorbar
    caxis([0,3e04])
    title('Range-Doppler heatmap')
end


end