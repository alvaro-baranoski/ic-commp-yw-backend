from get_data import get_data_from_api
from datetime import datetime
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import data_preprocessing as dpp
from statsmodels.regression.linear_model import yule_walker


def get_modes(processedFreq, fs, modelOrder=10):
    ar, sigma = yule_walker(processedFreq, order=modelOrder, method="mle")
    ar *= -1

    polyCoeff = np.array([1])
    polyCoeff = np.append(polyCoeff, ar)

    raizes_est_z = np.roots(polyCoeff)
    raizes_est_s = np.log(raizes_est_z) * fs

    # Remove negative frequencies
    raizes_est_s = [mode for mode in raizes_est_s if mode.imag > 0]

    # Calculates frequency in hertz and damping ratio in percentage
    freq_y = [mode.imag / (2 * np.pi) for mode in raizes_est_s]
    damp_x = [-np.divide(mode.real, np.absolute(mode))
              for mode in raizes_est_s]

    return damp_x, freq_y


def butterworth(data, cutoff, order, fs, kind="lowpass"):
    # highpass filter
    nyq = fs * 0.5

    cutoff = cutoff / nyq

    sos = signal.butter(order, cutoff, btype=kind, output="sos")

    filtrada = signal.sosfilt(sos, data)

    return filtrada


def correct_length(data, batch):
    # Corrects length of frequency list
    if len(data) % batch != 0:
        exactMult = np.floor(len(data) / batch)
        exactLen = int(exactMult * batch)
        lenDiff = len(data) - exactLen
        data = data[:-lenDiff]
    return data


print("starting program")

# Sampling rate in Hz
sampleRate = 15
# Set the data time window in minutes
timeWindow = 60
# Select PMU based on user input
pmuSelect = "eficiencia"

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

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array(
    [np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

######################### PARCEL CONFIG #########################

# Set size of data blocks in minutes
numberBlocks = 3

# Corrects length of frequency list
frequency = correct_length(freqValues, batch=numberBlocks)

# Instantiate list for output values
processedFreq = np.array([])

#freqValues = signal.decimate(freqValues, int(60/5))

######################### DATA PARCELING #########################
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

    dataBlock = butterworth(dataBlock, cutoff=0.3, order=16,
                            fs=sampleRate, kind="highpass")
    dataBlock = butterworth(dataBlock, cutoff=7.0, order=16,
                            fs=sampleRate, kind="lowpass")

    # Append processed data
    processedFreq = np.append(processedFreq, dataBlock)

######################### YULE-WALKER #########################
damp, freq = get_modes(processedFreq, fs=sampleRate, modelOrder=20)

######################### PLOT #########################
plt.scatter(freq, damp)
plt.ylabel("Fator de amortecimento")
plt.xlabel("Frequência [Hz]")
plt.show()
