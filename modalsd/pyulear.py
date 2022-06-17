import numpy as np
from scipy import signal
from statsmodels.regression.linear_model import yule_walker

def pyulear(x, order, nfft, fs):
    # From matlab
    # no exemplo, nfft = 2000 (igual a num_seg), fs = 20
    [pxx, freq] = arspectra(x, order, nfft, fs)


def arspectra(x, order, nfft, fs):
    ar, sigma = yule_walker(x, order=order, method="mle")
    ar *= -1
    ar = np.append(np.array([1]), ar)
    [w, h] = signal.freqz(1, ar, nfft, whole=True, fs=fs)
    return [1, 1]