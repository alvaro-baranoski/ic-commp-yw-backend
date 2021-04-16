#! /opt/ic-commp/bin/python3 startup.py

from get_data import get_data_from_api
from datetime import datetime
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as dpp
from statsmodels.regression.linear_model import yule_walker
from sys import argv
from json import dumps

# Sampling rate in Hz
sampleRate = int(argv[3])

# Set the data time window in minutes
timeWindow = int(argv[2])

# Select PMU based on user input
pmuSelect = argv[1]

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
apiData = np.array([get_data_from_api(startTime,
                                      endTime,
                                      feed_id=pmuSelect,
                                      interval=sampleRate,
                                      interval_type=1,
                                      skip_missing=0)])

# Splits data into time and frequency values and removes missing data
unixValues = np.array([i[0] for i in apiData[0]])
freqValues = np.array([i[1] for i in apiData[0]], dtype=np.float64)
freqValues_toPHP = np.array([i[1] for i in apiData[0]], dtype=np.float64)

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array(
    [np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

######################### FILTER DESIGN #########################

# FIR highpass filter coefficient design
highpassFreq = 0.15
hpCoef = np.float32(signal.firwin(numtaps=999,
                                  cutoff=highpassFreq,
                                  window='hann',
                                  pass_zero='highpass',
                                  fs=sampleRate))

######################### PARCEL CONFIG #########################

# Set size of data blocks in minutes
numberBlocks = timeWindow / 10

# Corrects length of frequency list
if len(freqValues) % numberBlocks != 0:
    exactMult = np.floor(len(freqValues) / numberBlocks)
    exactLen = int(exactMult * numberBlocks)
    lenDiff = len(freqValues) - exactLen
    freqValues = freqValues[:-lenDiff]

# Instantiate list for output values
processedFreq = np.array([])

######################### DATA PARCELING #########################
for dataBlock in np.array_split(freqValues, numberBlocks):

    # Check for long NaN runs
    nanRun = dpp.find_nan_run(dataBlock, run_max=10)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Detrend
    dataBlock -= np.nanmean(dataBlock)

    # HP filter
    dataBlock = signal.filtfilt(hpCoef, 1, dataBlock)

    # Outlier removal
    dataBlock = dpp.mean_outlier_removal(dataBlock, k=3.5)

    # Linear interpolation
    dataBlock = dpp.linear_interpolation(dataBlock)

    # Append processed data
    processedFreq = np.append(processedFreq, dataBlock)

######################### YULE-WALKER #########################
modelOrder = int(argv[4])
ar, sigma = yule_walker(processedFreq, order=modelOrder)
ar *= -1

polyCoeff = np.array([1])
polyCoeff = np.append(polyCoeff, ar)

raizes_est_z = np.roots(polyCoeff)
raizes_est_s = np.log(raizes_est_z) * sampleRate

# # Remove negative frequencies
raizes_est_s = [mode for mode in raizes_est_s if mode.imag > 0]

# Calculates frequency in hertz and damping ratio in percentage
freq_x = [mode.imag / (2 * np.pi) for mode in raizes_est_s]
damp_y = [-mode.real * 100 / np.absolute(mode) for mode in raizes_est_s]

######################### DATA SEND #########################

# Prepares dictionary for JSON file
data_to_php = {
    "freq": freqValues_toPHP.tolist(),
    "date": timeValues.astype(str).tolist(),
    "welch": damp_y,
    "welch_freq": freq_x
}

# Sends dict data to php files over JSON
print(dumps(data_to_php))
