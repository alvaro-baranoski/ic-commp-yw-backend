#! /opt/ic-commp/bin/python3 startup.py

from sys import argv, path
from crossvalidation import crossvalidation

# TODO: Ajustar nome do folder para igualar o do servidor
path.append("/opt/yulewalker/modalsd")
# path.append("..\ic-commp-yw-backend\modalsd")

from pyulear import pyulear
from modalsd import modalsd
from modalsd_3d import modalsd_3d

from get_data import get_data_from_api, get_data_from_welch
from datetime import datetime
import math
import numpy as np
import data_preprocessing as dpp
from json import dumps


# Select PMU based on user input
pmu_select = argv[1]
# Set the data time window in minutes
# Default value: 60
time_window = int(argv[2])
# Sampling rate in Hz
# Default value: 15
sample_rate = int(argv[3])
# Polynomial order
# Default value: 20
order = int(argv[4])
# Filter lower cutoff frequency
# Default value: 0.3
threshold_low = float(argv[5])
# Filter higher cutoff frequency
# Default value: 7.0
threshold_high = float(argv[6])
# Outlier detection constant
# Default value: 3.5
outlier_constant = float(argv[7])
# View type
viewSelect = argv[8]

FS_DOWN = 20
# Em s
WINDOW_TIME = 100
SLIDE = 120

pmu = 0

if pmu_select == "eficiencia":
    pmu = 506
elif pmu_select == "cabine":
    pmu = 515
elif pmu_select == "palotina":
    pmu = 524
elif pmu_select == "agrarias":
    pmu = 533

######################### DATE CONFIGURATION #########################

# Get time window from current time in unix milisseconds format
endTime = datetime.now()
endTime = int((endTime.timestamp() - 60) * 1000)
startTime = endTime - (time_window * 60 * 1000)

######################### DATA AQUISITION #########################

# Get the frequency data based on the start and end time
apiData = np.array([get_data_from_api(
    startTime,
    endTime,
    feed_id=pmu,
    interval=sample_rate,
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
signalff, ts1, fs1 = dpp.preprocessamento(
    freqValues, 
    ts, 
    fs, 
    FS_DOWN, 
    threshold_low, 
    threshold_high, 
    outlier_constant
)

######################### YULE-WALKER #########################
damp_basic, freq_basic = dpp.get_modes(signalff, fs=fs1, modelOrder=order)

######################### STAB DIAGRAM #########################
num_seg = fs1 * WINDOW_TIME
[pxx, freq] = pyulear(signalff, order, num_seg, fs1)

c_mpf, c_f, c_stab_freq_fn, c_stab_freq_mn, c_stab_fn, c_stab_mn, c_not_stab_fn, c_not_stab_mn = \
modalsd(pxx, freq, fs1, order, finish="return")

c_not_stab_fn = dpp.nan_to_none(c_not_stab_fn)

######################### 3D DIAGRAM #########################
d3_freq, d3_damp, main_modes = \
modalsd_3d(signalff, order, FS_DOWN, WINDOW_TIME, SLIDE, finish="return")

######################### CROSS VALIDATION ###################
welch_data = get_data_from_welch(
    pmu_select, 
    time_window, 
    sample_rate,
    threshold_low,
    threshold_high,
    outlier_constant)

main_modes = crossvalidation(main_modes, welch_data["peaks"])
######################### DATA SEND #########################

# Prepares dictionary for JSON file
data_to_php = {
    "freq": freqValues_toPHP.tolist(),
    "date": timeValues.astype(str).tolist(),
    "damp": damp_basic,
    "modes": freq_basic if type(freq_basic) == list else freq_basic.tolist(),
    "freq_process": signalff.tolist(),
    "c_mpf": c_mpf.tolist(),
    "c_f": c_f.tolist(),
    "c_stab_freq_fn": c_stab_freq_fn.tolist(),
    "c_stab_freq_mn": c_stab_freq_mn.tolist(),
    "c_stab_fn": c_stab_fn.tolist(),
    "c_stab_mn": c_stab_mn.tolist(),
    "c_not_stab_fn": c_not_stab_fn,
    "c_not_stab_mn": c_not_stab_mn.tolist(),
    "d3_freq": d3_freq.tolist(),
    "d3_damp": d3_damp.tolist(),
    "main_modes": main_modes,
    "view": viewSelect
}

# Sends dict data to php files over JSON
data_dump = dumps(data_to_php)
print(data_dump)
