#! /opt/ic-commp/bin/python3 startup.py

from sys import argv, path

# TODO: Ajustar nome do folder para igualar o do servidor
path.append("../yulewalker/modalsd")
# path.append("..\ic-commp-yw-backend\modalsd")

from pyulear import pyulear
from modalsd import modalsd
from modalsd_3d import modalsd_3d

from get_data import get_data_from_api
from datetime import datetime
import math
import numpy as np
import data_preprocessing as dpp
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
filtLowpass = float(argv[5])
# Filter higher cutoff frequency
# Default value: 7.0
filtHighpass = float(argv[6])
# Outlier detection constant
# Default value: 3.5
outlier_constant = float(argv[7])
# View type
viewSelect = argv[8]

FS_DOWN = 20
# Em s
WINDOW_TIME = 100


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

ts = (timeValues[2] - timeValues[1]) / np.timedelta64(1, 's')
fs = round(1 / ts)

######################### PRE PROCESSING #########################
signalff, ts1, fs1 = \
dpp.preprocessamento(freqValues, ts, fs, fsDown=FS_DOWN, filtLowpass=filtHighpass, k=3)

######################### YULE-WALKER #########################
damp, freq = dpp.get_modes(signalff, fs=sampleRate, modelOrder=order)


######################### DATA SEND #########################

# Prepares dictionary for JSON file
data_to_php = {
    "freq": freqValues_toPHP.tolist(),
    "date": timeValues.astype(str).tolist(),
    "damp": damp,
    "modes": freq if type(freq) == list else freq.tolist()
}

# Adds advanced view type
if (viewSelect == 'complete'):
    data_to_php["freq_process"] = signalff.tolist()

    ######################### STAB DIAGRAM #########################
    num_seg = fs1 * WINDOW_TIME
    [pxx, freq] = pyulear(signalff, order, num_seg, fs1)

    frf, f, mode_fn, mode_stab_fn, mode_stab_dr = \
    modalsd(pxx, freq, fs1, order, finish="return")

    data_to_php["frf"] = frf.tolist()
    data_to_php["frf_f"] = f.tolist()
    data_to_php["mode_fn"] = mode_fn
    data_to_php["mode_stab_fn"] = mode_stab_fn.tolist()
    data_to_php["mode_stab_dr"] = mode_stab_dr.tolist()

# Sends dict data to php files over JSON
data_dump = dumps(data_to_php)
print(data_dump)
