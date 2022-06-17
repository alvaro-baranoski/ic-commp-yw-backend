import numpy as np


def modalsd_3d(signalff, fs, window, slide):
    # TODO: DEPOIS CONFERIR COMO QUE CALCULA NO PROGRAMA
    signal_length = 1200

    num_windows = (signal_length - window)/slide + 1

    foo = 0
    stable_modes = np.zeros((0, 2))

    for i in range(int(num_windows)):
        
        if (foo == 0):
            signal_w = signalff[0:window*fs, :]
        else:
            signal_w = signalff[foo*fs:(window+foo)*fs, :]