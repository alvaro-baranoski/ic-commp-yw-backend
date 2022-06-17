import csv
import matplotlib.pyplot as plt
import numpy as np
from modalsd import modalsd
from pyulear import pyulear


##############################################
# ANALISANDO A FUNÇÃO MODAL_PLOT DO ARQUIVO 
# MAIN.PY DENTRO DE RESULTADOS_PRELIMINARES
##############################################
# Diagrama de estabilização convencional

freq = []
pxx = []
fs = 20

with open("modalsd\data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        freq.append(float(row[0]))
        pxx.append(float(row[1]))

pxx = np.array(pxx, dtype=np.float64)
freq = np.array(freq, dtype=np.float64)

modalsd(frf=pxx, f=freq, fs=fs)


# Diagrama de estabilização 3D
signalff = []

with open("modalsd\signalff.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        signalff.append(float(row[0]))

signalff = np.array(signalff, dtype=np.float64)

[pxx, freq] = pyulear(signalff, order=25, nfft=2000, fs=fs)

plt.plot(freq, pxx)
plt.show()

pass