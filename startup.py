#! /opt/ic-commp/bin/python3 startup.py

from get_data import get_data_from_api
from datetime import datetime
from scipy import signal
import math
import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as dpp
from statsmodels.regression.linear_model import yule_walker
from sys import argv
from json import dumps

# Select PMU based on user input
pmuSelect = argv[1]
# Set the data time window in minutes
# Default value: 60
timeWindow = int(argv[2])
# Sampling rate in Hz
# Default value: 15
sampleRate = int(argv[3])
# Polynomial order
# Default value: 20
order = int(argv[4])
# Filter lower cutoff frequency
# Default value: 0.3
lower_filter = float(argv[5])
# Filter higher cutoff frequency
# Default value: 7.0
higher_filter = float(argv[6])
# Outlier detection constant
# Default value: 3.5
outlier_constant = float(argv[7])
# View type
viewSelect = argv[8]

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
apiData = np.array([get_data_from_api(
    startTime,
    endTime,
    feed_id=pmuSelect,
    interval=sampleRate,
    interval_type=1,
    skip_missing=0
)])

# Splits data into time and frequency values and removes missing data
unixValues = np.array([i[0] for i in apiData[0]])
freqValues = np.array([i[1] for i in apiData[0]], dtype=np.float64)

# Pads NaN with linear interpolation
# (this step is required for avoiding JSON parsing bug)
freqValues_toPHP = np.array([i[1] for i in apiData[0]], dtype=np.float64)
freqValues_toPHP = dpp.linear_interpolation(freqValues_toPHP)

# Checa se valores de frequência estão disponíveis
if (all(math.isnan(v) for v in freqValues)):
    raise NameError('Dados da PMU indisponíveis')

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array(
    [np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

######################### PARCEL CONFIG #########################

# Set size of data blocks in minutes
numberBlocks = 3

# Corrects length of frequency list
frequency = dpp.correct_length(freqValues, batch=numberBlocks)

# Instantiate list for output values
processedFreq = np.array([])

######################### DATA PARCELING #########################
for dataBlock in np.array_split(freqValues, numberBlocks):

    # Check for long NaN runs
    nanRun = dpp.find_nan_run(dataBlock, run_max=10)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Outlier removal
    dataBlock = dpp.mean_outlier_removal(dataBlock, k=outlier_constant)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Detrend
    dataBlock -= np.nanmean(dataBlock)

    dataBlock = dpp.butterworth(
        dataBlock, 
        cutoff=lower_filter, 
        order=16,
        fs=sampleRate, 
        kind="highpass"
    )

    dataBlock = dpp.butterworth(
        dataBlock, 
        cutoff=higher_filter, 
        order=16,
        fs=sampleRate, 
        kind="lowpass"
    )

    # Append processed data
    processedFreq = np.append(processedFreq, dataBlock)

######################### YULE-WALKER #########################
damp, freq = dpp.get_modes(processedFreq, fs=sampleRate, modelOrder=order)

######################### DATA SEND #########################

# Prepares dictionary for JSON file
data_to_php = {
    "freq": freqValues_toPHP.tolist(),
    "date": timeValues.astype(str).tolist(),
    "damp": damp,
    "modes": freq
}

# Adds advanced view type
if (viewSelect == 'complete'):
    data_to_php["freq_process"] = processedFreq.tolist()

# Sends dict data to php files over JSON
print(dumps(data_to_php))
