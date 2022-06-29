#! /opt/ic-commp/bin/python3 startup.py

from src.main.get_data import get_data_from_api
from datetime import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import src.main.data_preprocessing as dpp
from sys import argv, path

path.append("modalsd")

from pyulear import pyulear
from modalsd import modalsd
from modalsd_3d import modalsd_3d

# Select PMU based on user input
pmuSelect = 533
# Set the data time window in minutes
# Default value: 60
timeWindow = 20
# Sampling rate in Hz
# Default value: 15
sampleRate = 100
# Polynomial order
# Default value: 20
order = 23
# Filter lower cutoff frequency
# Default value: 0.3
filtLowpass = 0.07
# Filter higher cutoff frequency
# Default value: 7.0
filtHighpass = 4.0
# Outlier detection constant
# Default value: 3.5
outlierConstant = 5
# View type
viewSelect = "complete"

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

# Checa se valores de frequência estão disponíveis
if (all(math.isnan(v) for v in freqValues)):
    raise NameError('Dados da PMU indisponíveis')

# Converts unix time to Numpy DateTime64 time milisseconds and converts from GMT time to local time
timeValues = np.array(
    [np.datetime64(int(i - (3 * 3600000)), 'ms') for i in unixValues])

ts = (timeValues[2] - timeValues[1]) / np.timedelta64(1, 's')
fs = round(1 / ts)

######################### PARCEL CONFIG #########################
signalff, ts1, fs1 = \
dpp.preprocessamento(freqValues, ts, fs, fsDown=FS_DOWN, filtLowpass=filtHighpass, k=3)

plt.plot(signalff)
plt.show()

######################### YULE-WALKER #########################
num_seg = fs1 * WINDOW_TIME
[pxx, freq] = pyulear(signalff, order, num_seg, fs1)

frf, f, mode_fn, mode_stab_fn, mode_stab_dr = \
    modalsd(pxx, freq, fs1, order, finish="return")
######################### DATA SEND #########################