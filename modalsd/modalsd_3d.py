from pyulear import pyulear
import numpy as np
from modalsd import modalsd
import matplotlib.pyplot as plt

def modalsd_3d(signalff, order, fs, window_time, slide):
    num_seg = fs * window_time

    # TODO: DEPOIS CONFERIR COMO QUE CALCULA NO PROGRAMA
    signal_length = 1200
    window = 10*60 # em segundos

    num_windows = (signal_length - window)/slide + 1

    foo = 0
    stable_modes = np.zeros((0, 2))

    for i in range(int(num_windows)):
        
        if (foo == 0):
            signal_w = signalff[0:window*fs]
        else:
            signal_w = signalff[foo*fs:(window+foo)*fs]

        [pxx, freq] = pyulear(signal_w, order, num_seg, fs)

        [fn, dr] = modalsd(frf=pxx, f=freq, fs=fs, max_modes=25, conventional=False)

        index = np.where(fn > 0)

        fn_s = fn[index]
        dr_s = 100 * dr[index]

        stable_modes = np.append(stable_modes, np.column_stack((fn_s, dr_s)), axis=0)
        
        foo += slide

    # Histogram plot
    MIN_FREQ = 0
    MAX_FREQ = 2.5
    MIN_DAMP = 0
    MAX_DAMP = 20
    TICKS_FREQ = 21

    plt.hist2d(stable_modes[:,0], stable_modes[:,1], bins=[150, 150])
    plt.xlim(MIN_FREQ, MAX_FREQ)
    plt.ylim(MIN_DAMP, MAX_DAMP)
    plt.xticks(np.arange(MIN_FREQ, MAX_FREQ, TICKS_FREQ))
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Número de ocorrências')
    plt.title('Histograma de frequências')
    plt.grid(True)
    plt.show()

    pass