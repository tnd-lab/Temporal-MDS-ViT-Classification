% specify data name and load data as variable data_frames

function [save_det_data, Dopdata_sum, Angdata_crop, center_r, center_a] = dfft(seq_path)

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
set_frame_number = 30;
frame_start = 1;
frame_end = set_frame_number;
Is_Windowed = 1;% 1==> Windowing before doing range and angle fft
M = 16; % number of frames for generating micro-Doppler image
Lr = 11; % length of cropped region along range
La = 5; % length of cropped region along angle
Ang_seq = [2,5,8,11,14]; % dialated angle bin index for cropping
veloc_bin_norm = 2; % velocity normaliztion term for DBSCAN
dis_thrs = [20, 16, 20]; % range_thrs, veloc_thrs, angle_thrs for DBSCAN
WINDOW =  255; % STFT parameters
NOVEPLAP = 240; % STFT parameters

% Function to process radar data and generate visualizations
% Input:
%   seq_dir: Path to the .mat file containing radar raw frames
% Output:
%   save_det_data: Processed detection data in the specified format

% Load data
load(seq_path); % Assumes the variable `data_frames` is loaded

% Arrange data for each chirp
data_frame = permute(adcData, [3, 1, 2, 4]);
data_frame = reshape(data_frame, 4, []);
data_chirp = [];
for cj = 1:Tx * loop
    temp_data = data_frame(:, (cj - 1) * samples + 1 : cj * samples);
    data_chirp(:, :, cj) = temp_data;
end

% Separate odd-index and even-index chirps for TDM-MIMO
chirp_odd = data_chirp(:, :, 1:2:end);
chirp_even = data_chirp(:, :, 2:2:end);

% Permute to [samples, Rx, chirp]
chirp_odd = permute(chirp_odd, [2, 1, 3]);
chirp_even = permute(chirp_even, [2, 1, 3]);

% Range FFT
Rangedata_odd = fft_range(chirp_odd, fft_Rang, Is_Windowed);
Rangedata_even = fft_range(chirp_even, fft_Rang, Is_Windowed);

% Doppler FFT
Dopplerdata_odd = fft_doppler(Rangedata_odd, fft_Vel, 0);
Dopplerdata_even = fft_doppler(Rangedata_even, fft_Vel, 0);
Dopdata_sum = squeeze(mean(abs(Dopplerdata_odd), 2));

% CFAR detection
Pfa = 1e-4; % Probability of false alarm
Resl_indx = cfar_RV(Dopdata_sum, fft_Rang, num_crop, Pfa);
detout = peakGrouping(Resl_indx);

% Doppler compensation
for ri = num_crop + 1 : fft_Rang - num_crop
    find_idx = find(detout(2, :) == ri);
    if isempty(find_idx)
        continue
    else
        pick_idx = find_idx(1); % Pick the first larger velocity
        pha_comp_term = exp(-1i * pi * (detout(1, pick_idx) - fft_Vel / 2 - 1) / fft_Vel);
        Rangedata_even(ri, :, :) = Rangedata_even(ri, :, :) * pha_comp_term;
    end
end

Rangedata_merge = [Rangedata_odd, Rangedata_even];

% Angle FFT
Angdata = fft_angle(Rangedata_merge, fft_Ang, Is_Windowed);
Angdata_crop = Angdata(num_crop + 1 : fft_Rang - num_crop, :, :);
Angdata_crop = Normalize(Angdata_crop, max_value);

% Angle estimation for detected point clouds
Dopplerdata_merge = permute([Dopplerdata_odd, Dopplerdata_even], [2, 1, 3]);
[Resel_agl, ~, rng_excd_list] = angle_estim_dets(detout, Dopplerdata_merge, fft_Vel, ...
    fft_Ang, Rx, Tx, num_crop);

% Transform bin index to range/velocity/angle
Resel_agl_deg = agl_grid(1, Resel_agl)';
Resel_vel = vel_grid(detout(1, :), 1);
Resel_rng = rng_grid(detout(2, :), 1);

% Save detection data
save_det_data = [detout(2, :)', detout(1, :)', Resel_agl', detout(3, :)', ...
    Resel_rng, Resel_vel, Resel_agl_deg];

% Filter out points in the cropped region
if ~isempty(rng_excd_list)
    save_det_data(rng_excd_list, :) = [];
end



% DBSCAN clustering
if ~isempty(rng_excd_list)
    dets_cluster = clustering(save_det_data, fft_Vel, veloc_bin_norm, ...
        dis_thrs, rng_grid, agl_grid);
end

% determine the center for cropping region
center_r = dets_cluster(1, 1); % range center for cropped region
center_a = dets_cluster(1, 3); % angle center for cropped region
    

end
