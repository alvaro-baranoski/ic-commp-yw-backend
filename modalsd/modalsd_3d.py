from pyulear import pyulear
import numpy as np
from modalsd import modalsd
import matplotlib.pyplot as plt

# Histogram plot
# TODO: Alinhar valores com frontend
FREQ_MIN = 0
FREQ_MAX = 2.5
FREQ_SIZE = 0.1

DAMP_MIN = 0
DAMP_MAX = 50
DAMP_SIZE = 1

freq_bin = int((FREQ_MAX - FREQ_MIN) / FREQ_SIZE)
damp_bin = int((DAMP_MAX - DAMP_MIN) / DAMP_SIZE)

def modalsd_3d(signalff, order, fs, window_time, slide, finish = "plot"):
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

        [fn, dr] = modalsd(frf=pxx, f=freq, fs=fs, max_modes=order, finish="3d")

        index = np.where(fn > 0)

        fn_s = fn[index]
        dr_s = 100 * dr[index]

        stable_modes = np.append(stable_modes, np.column_stack((fn_s, dr_s)), axis=0)
        
        foo += slide

    main_modes = get_main_modes(stable_modes)

    if (finish == "return"):
        return stable_modes[:, 0], stable_modes[:, 1], main_modes


    plt.hist2d(
        stable_modes[:,0], 
        stable_modes[:,1], 
        bins=[freq_bin, damp_bin],
        range=[[FREQ_MIN, FREQ_MAX], [DAMP_MIN, DAMP_MAX]])

    plt.xlim(FREQ_MIN, FREQ_MAX)
    plt.ylim(DAMP_MIN, DAMP_MAX)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Número de ocorrências')
    plt.title('Histograma de frequências')
    plt.grid(True)
    plt.show()



def get_main_modes(stable_modes):
    '''
    Calcula os dois modos mais incidentes calculados
    '''

    [h, xedges, yedges] = np.histogram2d(
        stable_modes[:,0], 
        stable_modes[:,1],
        bins=[freq_bin, damp_bin],
        range=[[FREQ_MIN, FREQ_MAX], [DAMP_MIN, DAMP_MAX]])

    main_modes = []

    # Calcula os dois modos principais presentes
    ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    mode = {
        "presence": h[ind],
        "freq_interval": [round(xedges[ind[0]], 2), round(xedges[ind[0] + 1], 2)],
        "damp_interval": [round(yedges[ind[1]], 2), round(yedges[ind[1] + 1], 2)]
    }
    main_modes.append(mode)

    h[ind] = 0

    ind = np.unravel_index(np.argmax(h, axis=None), h.shape)
    mode = {
        "presence": h[ind],
        "freq_interval": [round(xedges[ind[0]], 2), round(xedges[ind[0] + 1], 2)],
        "damp_interval": [round(yedges[ind[1]], 2), round(yedges[ind[1] + 1], 2)]
    }
    main_modes.append(mode)

    return main_modes