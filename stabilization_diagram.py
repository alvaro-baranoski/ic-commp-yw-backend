from ast import dump
from ctypes import sizeof
from get_data import get_data_from_api
from datetime import datetime
from scipy import signal
import math
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

# Calculo do filtro passa faixa
f1, f2 = 0.1, 2
coef = signal.firwin(
    numtaps=500,
    cutoff=[f1, f2],
    window='hann',
    pass_zero=False,
    fs=sampleRate
)

# Interpolação linear
interpoled = dpp.linear_interpolation(freqValues)

# Outlier removal
interpoled = dpp.mean_outlier_removal(interpoled, k=3.5)

# Linear interpolation
interpoled = dpp.linear_interpolation(interpoled)

# Aplicação do filtro
filtered = signal.filtfilt(coef, 1, interpoled)

# Downsample
downsample_freq = 5
downsampled = signal.decimate(filtered, int(sampleRate/downsample_freq))

# PERGUNTA: COMO COMPARAR OS RESULTADOS OBTIDOS, JÁ QUE É UMA LISTA COM TAMANHOS
# DIFERENTES? 

# Duração da janela em segundos
window_duration = 10 * 60
# Movimentação da janela em segundos
window_moving_duration = 10
# Range de ordens
higher_order = 25
lower_order = 2

window_points = window_duration * downsample_freq
window_moving_points = window_moving_duration * downsample_freq

lower_index = 0
higher_index = window_points

freq_list = []
damp_list = []

prev_freq = []
prev_damp = []

# for i in range(30):
batch = downsampled[lower_index:higher_index]
for order in range(higher_order, lower_order-1, -1):
    print(f'Ordem número: {order}')
    damp, freq = get_modes(batch, fs=downsample_freq, modelOrder=order)
    if (order == higher_order):
        prev_freq = freq
        prev_damp = damp
        continue

    

    # print('comparação de frequências: ')
    # print(prev_freq)
    # print(freq)

    # print('Comparação de amortecimentos: ')
    # print(prev_damp)
    # print(damp)

    prev_freq = freq
    prev_damp = damp
    
    freq_list.extend(freq)
    damp_list.extend(damp)


# lower_index += window_moving_points
# higher_index += window_moving_points

# plt.plot(batch)
plt.scatter(freq_list, damp_list)
plt.show()

