% plot 3D point clouds
function [axh] = plot_pointclouds(detout, save_path)
% detout format: % [range bin, velocity bin, angle bin, power, range(m), ...
% velocity (m/s), angle(degree)]
if nargin < 2
   save_path = [];
end

if ~isempty(save_path)
    fig = figure('visible','off');
    % x-direction: Doppler, y-direction: angle, z-direction: range
    [axh] = scatter3(detout(:, 6), detout(:, 7), detout(:, 5), 'filled');
    xlabel('Doppler velocity (m/s)')
    ylabel('Azimuth angle (degrees)')
    zlabel('Range (m)')
    axis([-5, 5 -60 60 2 25]);
    title('3D point clouds')
    grid on
    
    % Save the figure if a path is provided
    if nargin > 1 && ~isempty(save_path)
        [folder, ~, ~] = fileparts(save_path);
        if ~isempty(folder) && ~isfolder(folder)
            mkdir(folder); % Create the directory if it doesn't exist
        end
        saveas(fig, save_path); % Save the figure in the specified format
    end
else
    fig = figure('visible','on');
    % x-direction: Doppler, y-direction: angle, z-direction: range
    [axh] = scatter3(detout(:, 6), detout(:, 7), detout(:, 5), 'filled');
    xlabel('Doppler velocity (m/s)')
    ylabel('Azimuth angle (degrees)')
    zlabel('Range (m)')
    axis([-5, 5 -60 60 2 25]);
    title('3D point clouds')
    grid on
end

end

