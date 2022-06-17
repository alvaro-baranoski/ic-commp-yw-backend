import numpy as np
from scipy import signal
import spectrum

def pyulear(x, order, nfft, fs):
    # From matlab
    # no exemplo, nfft = 2000 (igual a num_seg), fs = 20
    [pxx, freq] = arspectra(x, order, nfft, fs)
    return pxx, freq


def arspectra(x, order, nfft, fs):
    [a, p, k] = spectrum.aryule(x, order)
    a = np.append(np.array([1]), a)
    [w, h] = signal.freqz(1, a, nfft, whole=True, fs=fs)
    sxx = np.abs(h)**2 * p

    [pxx, w] = computepsd(sxx, w, nfft, fs)

    return pxx, w


def computepsd(sxx, w, nfft, fs):
    # ODD
    # TODO: COMPARAR E COMPLETAR
    if np.remainder(nfft, 2):
        select = range(0, (nfft+1)/2)
        sxx_unscaled = sxx[select, :]

    # EVEN
    else:
        select = np.arange(0, nfft/2 + 1, dtype=int)
        sxx_unscaled = sxx[select]
        sxx = np.append(sxx_unscaled[0], 2 * sxx_unscaled[1:-1])
        sxx = np.append(sxx, sxx_unscaled[-1])

    w = w[select]

    # Compute the PSD [Power/freq]
    pxx = sxx / fs
    
    return pxx, w