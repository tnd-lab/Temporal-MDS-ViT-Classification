clc
clearvars
close all

% Define the root directory of the dataset
dataset_root = './template data/Automotive/';

% Get list of all subdirectories
main_folders = dir(dataset_root);

% parameter setting
params = get_params_value();
% constant parameters
c = params.c; % Speed of light in air (m/s)
fc = params.fc; % Center frequency (Hz)
lambda = params.lambda;
Rx = params.Rx;
Tx = params.Tx;

% configuration parameters
Fs = params.Fs;
sweepSlope = params.sweepSlope;
samples = params.samples;
loop = params.loop;

Tc = params.Tc; % us 
fft_Rang = params.fft_Rang;
fft_Vel = params.fft_Vel;
fft_Ang = params.fft_Ang;
num_crop = params.num_crop;
max_value = params.max_value; % normalization the maximum of data WITH 1843

% Creat grid table
rng_grid = params.rng_grid;
agl_grid = params.agl_grid;
vel_grid = params.vel_grid;

% Algorithm parameters
data_each_frame = samples*loop*Tx;
Is_Windowed = 1;% 1==> Windowing before doing range and angle fft
M = 16; % number of frames for generating micro-Doppler image
Lr = 11; % length of cropped region along range
La = 5; % length of cropped region along angle
Ang_seq = [2,5,8,11,14]; % dialated angle bin index for cropping
veloc_bin_norm = 2; % velocity normaliztion term for DBSCAN
dis_thrs = [20, 16, 20]; % range_thrs, veloc_thrs, angle_thrs for DBSCAN
WINDOW =  255; % STFT parameters
NOVEPLAP = 240; % STFT parameters


% Loop through each entry in the directory
for i = 1:length(main_folders)
    % Get the name of the folder or file
    folder_name = main_folders(i).name;

    % Create the full path for the subfolder
    subfolder_path = fullfile(dataset_root, folder_name);

    fprintf('Processing folder: %s\n', subfolder_path);
    
    % Get list of files in the subfolder
    files = dir(fullfile(subfolder_path, 'radar_raw_frame')); % You can specify a pattern, e.g., '*.txt'


    % Loop through files in the current subfolder
    radarcube_crop = [];
    end_frame = M;
    count = 0;
    for j = 1:length(files)
        count = count + 1;
        file_name = files(j).name;

        % Skip '.' and '..'
        if strcmp(file_name, '.') || strcmp(file_name, '..')
            fprintf(file_name);
            continue;
        end

        % Full path to the file
        file_path = fullfile(strcat(subfolder_path, '/radar_raw_frame'), file_name);
        % Remove the '.mat' extension
        save_dir = erase(strrep(file_path, 'Automotive', 'Data'), '.mat');
        [save_det_data, Dopdata_sum, Angdata_crop, center_r, center_a] = dfft(file_path);

        % Plot range-angle image
        plot_rangeAng(Angdata_crop, rng_grid(num_crop + 1 : fft_Rang - num_crop), agl_grid, strcat(save_dir, '/range_angle.png'));
         
        % Plot 3D point clouds
        plot_pointclouds(save_det_data, strcat(save_dir, '/point_cloud.png'));
        
        % Plot range-Doppler image
        plot_rangeDop(Dopdata_sum, rng_grid, vel_grid, strcat(save_dir, '/range_doppler.png'));


        save(strcat(save_dir, '/Angdata_crop.mat'), 'Angdata_crop');
        save(strcat(save_dir, '/Dopdata_sum.mat'), 'Dopdata_sum');
        
        radarcube_crop = cat(3, radarcube_crop, Angdata_crop(center_r-(Lr-1)/2:center_r+(Lr-1)/2, center_a-(max(Ang_seq)/2+1)+Ang_seq, :));
        if count == end_frame
            %% STFT processing for generating microDoppler map
            data_conca = [];
            STFT_data = [];
            
            % reshae data to the formta [rangebin*anglebin, frames]
            for j = 1:Lr
                for i = 1:La
                    data_conca = [data_conca; squeeze(radarcube_crop(j,i,:))'];
                end
            end
            
            % STFT operation
            for h = 1:Lr*La
                [S,F,T] = spectrogram(data_conca(h,:), WINDOW, NOVEPLAP, 256, 1/Tc,'centered');
                v_grid_new = F*lambda/2;
                STFT_data = cat(3, STFT_data, S);
            end
            radarcube_crop = [];
            end_frame = end_frame + M;
            
            save(strcat(save_dir, '/STFT_data.mat'), 'STFT_data');

            
        end
    end

end
