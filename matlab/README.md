# Simulation MDS-Data

![image](/images/fig_mds_process.png)
## Prerequistes
To run matlab code, make sure run these commands
```
cd matlab
addpath(genpath('./'))
```

## Simulation
- To generate Range-Doppler, Range-Angle heatmap run: `generate_ra_3dfft.m`
- To generate MDS data run: `generate_microdoppler_stft.m`
- To generate the MDS data for training run: `getdata_mds.m`


## Acknowlegment
This simulation is based on great opensourced codebased and dataset

* [mmWave-radar-signal-processing-and-microDoppler-classification](https://github.com/Xiangyu-Gao/mmWave-radar-signal-processing-and-microDoppler-classification)

* [raw_ADC_radar_dataset_for_automotive_object_detection](https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection)
