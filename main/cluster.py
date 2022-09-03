#! /opt/ic-commp/bin/python3 startup.py

from json import dumps
from sys import argv, path

# TODO: Ajustar nome do folder para igualar o do servidor
path.append("/opt/yulewalker/modalsd")
# path.append("..\ic-commp-yw-backend\modalsd")

from modalsd_3d import modalsd_3d

from get_data import get_data_from_api
from datetime import datetime
import math
import numpy as np
import data_preprocessing as dpp

# Set the data time window in minutes
# Default value: 20
timeWindow = 20
# Sampling rate in Hz
# Default value: 100
sampleRate = 100
# Polynomial order
# Default value: 20
order = 20
# Filter lower cutoff frequency
# Default value: 0.3
threshold_low = 0.3
# Filter higher cutoff frequency
# Default value: 7.0
threshold_high = 7.0
# Outlier detection constant
# Default value: 3.5
outlier_constant = 5

FS_DOWN = 20
WINDOW_TIME = 100 # Em s
SLIDE = 120

# Eficiência, Cabine, Agrarias
pmuSelect = [506, 515, 533]

######################### DATE CONFIGURATION #########################

# Get time window from current time in unix milisseconds format
endTime = datetime.now()
endTime = int((endTime.timestamp() - 60) * 1000)
startTime = endTime - (timeWindow * 60 * 1000)

frequencies_array = []

######################### DATA AQUISITION #########################
for pmu in pmuSelect:
    # Get the frequency data based on the start and end time
    apiData = np.array([get_data_from_api(
        startTime,
        endTime,
        feed_id=pmu,
        interval=sampleRate,
        interval_type=1,
        skip_missing=0
    )])

    # Splits data into time and frequency values and removes missing data
    unixValues = np.array([i[0] for i in apiData[0]])
    freqValues = np.array([i[1] for i in apiData[0]], dtype=np.float64)

    # Checa se valores de frequência estão disponíveis
    if (all(math.isnan(v) for v in freqValues)):
        raise NameError('Dados da PMU indisponíveis')

    # Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
    timeValues = np.array(
        [np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

    ts = (timeValues[2] - timeValues[1]) / np.timedelta64(1, 's')
    fs = round(1 / ts)

    # Adds frequency to column
    frequencies_array.append(freqValues)

######################### PRE PROCESSING #########################

signals_array = []

for frequency in frequencies_array:
    signalff, ts1, fs1 = \
    dpp.preprocessamento(frequency, ts, fs, FS_DOWN, threshold_low, threshold_high, k=3)

    signals_array.append(signalff)

######################### 3D DIAGRAM #########################

modes_array = []

for signal in signals_array:
    d3_freq, d3_damp, main_modes = \
    modalsd_3d(signal, order, FS_DOWN, WINDOW_TIME, SLIDE, finish="return")

    modes_array.append(main_modes)

######################### FINAL MODES #########################

# result_modes = []
result_modes = np.array(modes_array).flatten().tolist()

for mode in result_modes:
    for other_mode in result_modes:
        isSameMode = mode == other_mode
        isSameFreq = mode["freq_interval"] == other_mode["freq_interval"]
        isSameDamp = mode["damp_interval"] == other_mode["damp_interval"]
        if (isSameFreq and isSameDamp and not isSameMode):
            mode["presence"] += other_mode["presence"]
            result_modes.remove(other_mode)

result_modes = sorted(result_modes, key=lambda d: d["presence"])
result_modes.reverse()

data_to_php = {
    "main_modes": result_modes
}

# Sends dict data to php files over JSON
data_dump = dumps(data_to_php)
print(data_dump)
