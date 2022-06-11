import csv
import matplotlib.pyplot as plt
import numpy as np
from modalsd import modalsd

##############################################
# ANALISANDO A FUNÇÃO MODAL_PLOT DO ARQUIVO 
# MAIN.PY DENTRO DE RESULTADOS_PRELIMINARES
##############################################

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