from ast import dump
from ctypes import sizeof
from src.main.get_data import get_data_from_api
from datetime import datetime
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt
import src.main.data_preprocessing as dpp
from statsmodels.regression.linear_model import yule_walker


print("starting program")

# Sampling rate in Hz
sampleRate = 15
# Set the data time window in minutes
timeWindow = 120
# Select PMU based on user input
pmuSelect = "palotina"

if pmuSelect == "eficiencia":
    pmuSelect = 506
elif pmuSelect == "cabine":
    pmuSelect = 515
elif pmuSelect == "palotina":
    pmuSelect = 524
elif pmuSelect == "agrarias":
    pmuSelect = 533

######################### DATE CONFIGURATION #########################

# Get time window from current time in unix milisseconds format
endTime = datetime.now()
endTime = int((endTime.timestamp() - 60) * 1000)
startTime = endTime - (timeWindow * 60 * 1000)

######################### DATA AQUISITION #########################

# Get the frequency data based on the start and end time
apiData = np.array(
    [get_data_from_api(
    startTime,
    endTime,
    feed_id=pmuSelect,
    interval=sampleRate,
    interval_type=1,
    skip_missing=0)])

# Splits data into time and frequency values and removes missing data
unixValues = np.array([i[0] for i in apiData[0]])
freqValues = np.array([i[1] for i in apiData[0]], dtype=np.float64)

if (all(math.isnan(v) for v in freqValues)):
    raise NameError('Dados da PMU indisponíveis')

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array([np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

######################### PRE PROCESSING #########################

# Set size of data blocks in minutes
numberBlocks = 3

# Corrects length of frequency list
frequency = dpp.correct_length(freqValues, batch=numberBlocks)

# Instantiate list for output values
processedFreq = np.array([])

for dataBlock in np.array_split(freqValues, numberBlocks):

    # Check for long NaN runs
    nanRun = dpp.find_nan_run(dataBlock, run_max=10)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Outlier removal
    dataBlock = dpp.mean_outlier_removal(dataBlock, k=3.5)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Detrend
    dataBlock -= np.nanmean(dataBlock)

    dataBlock = dpp.butterworth(
        dataBlock, 
        cutoff=0.3, 
        order=16,
        fs=sampleRate, 
        kind="highpass"
    )

    dataBlock = dpp.butterworth(
        dataBlock, 
        cutoff=7, 
        order=16,
        fs=sampleRate, 
        kind="lowpass"
    )

    # Append processed data
    processedFreq = np.append(processedFreq, dataBlock)

######################### STABILIZATION DIAGRAM #########################

# Duração da janela em segundos
window_duration = 10 * 60
# Movimentação da janela em segundos
window_moving_duration = 10
# Range de ordens
higher_order = 25
lower_order = 2
# Stabilization diagram thresholds (in %)
FREQ_LIMIT = 1.5
DAMP_LIMIT = 3.5

window_points = window_duration * sampleRate
window_moving_points = window_moving_duration * sampleRate

lower_index = 0
higher_index = window_points

freq_list = []
damp_list = []

prev_freq = []
prev_damp = []

# for i in range(30):
# batch = downsampled[lower_index:higher_index]
batch = processedFreq

# Gets the first mode list
prev_modes = []
damp, freq = dpp.get_modes(batch, fs=sampleRate, modelOrder=higher_order)
for i in range(len(freq)):
    prev_modes.append((freq[i], damp[i]))
prev_modes.sort(key=lambda x:x[0])

# Loop through the order range
for order in range(higher_order-1, lower_order-1, -1):
    current_modes = []
    print(f'-----------------------------------')
    print(f'Ordem número: {order}')
    
    damp, freq = dpp.get_modes(batch, fs=sampleRate, modelOrder=order)
    
    for i in range(len(freq)):
        current_modes.append((freq[i], damp[i]))
    
    current_modes.sort(key=lambda x:x[0])
    
    # Do some math
    # Loop through modes
    for i in range(len(current_modes)):
        # Find the closest frequency pair
        mode_n = current_modes[i]
        mode_n_1 = min(prev_modes, key=lambda x : abs(x[0] - mode_n[0]))
        
        freq_n = mode_n[0]
        damp_n = mode_n[1]
        freq_n_1 = mode_n_1[0]
        damp_n_1 = mode_n_1[1]
        
        is_freq_valid = abs((freq_n - freq_n_1)/freq_n) * 100 <= FREQ_LIMIT
        is_damp_valid = abs((damp_n - damp_n_1)/damp_n) * 100 <= DAMP_LIMIT

        if (is_freq_valid and is_damp_valid):
            print(f'{mode_n}   |   {mode_n_1}')


    prev_modes = current_modes


# lower_index += window_moving_points
# higher_index += window_moving_points

# plt.plot(batch)
# plt.scatter(freq_list, damp_list)
# plt.show()

