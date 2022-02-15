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
    damp_x = [-np.divide(mode.real, np.absolute(mode)) for mode in raizes_est_s]

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
sampleRate = 30
# Set the data time window in minutes
timeWindow = 120
# Select PMU based on user input
pmuSelect = "agrarias"

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

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array([np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])


f1, f2 = 0.1, 2
coef = signal.firwin(
    numtaps=500,
    cutoff=[f1, f2],
    window='hann',
    pass_zero=False,
    fs=sampleRate
)

interpoled = dpp.linear_interpolation(freqValues)

filtered = signal.filtfilt(coef, 1, interpoled)

downsample_freq = 5
downsampled = signal.decimate(filtered, int(sampleRate/downsample_freq))

# Duração da janela em segundos
window_duration = 10 * 60
# Movimentação da janela em segundos
window_moving = 10

window_points = window_duration * downsample_freq

batch = downsampled[:window_points]

# damp, freq = get_modes(downsampled, fs=downsample_freq, modelOrder=25)

# plt.scatter(freq, damp)
plt.plot(batch)
plt.show()
######################### PARCEL CONFIG #########################

# Set size of data blocks in minutes
# numberBlocks = 3

# Corrects length of frequency list
# frequency = correct_length(freqValues, batch=numberBlocks)

# Instantiate list for output values
# processedFreq = np.array([])

#freqValues = signal.decimate(freqValues, int(60/5))

######################### DATA PARCELING #########################
# for dataBlock in np.array_split(freqValues, numberBlocks):

#     # Check for long NaN runs
#     nanRun = dpp.find_nan_run(dataBlock, run_max=10)

#     # Linear interpolation
#     dataBlock = dpp.linear_interpolation(dataBlock)

#     # Outlier removal
#     dataBlock = dpp.mean_outlier_removal(dataBlock, k=3.5)

#     # Linear interpolation
#     dataBlock = dpp.linear_interpolation(dataBlock)

#     # Detrend
#     dataBlock -= np.nanmean(dataBlock)

#     dataBlock = butterworth(dataBlock, cutoff=0.3, order=16,
#                             fs=sampleRate, kind="highpass")
#     dataBlock = butterworth(dataBlock, cutoff=7.0, order=16,
#                             fs=sampleRate, kind="lowpass")

#     # Append processed data
#     processedFreq = np.append(processedFreq, dataBlock)

######################### YULE-WALKER #########################
# damp, freq = get_modes(processedFreq, fs=sampleRate, modelOrder=20)

# ######################### PLOT #########################
# plt.scatter(freq, damp)
# plt.ylabel("Fator de amortecimento")
# plt.xlabel("Frequência [Hz]")
# plt.show()
