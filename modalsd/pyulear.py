import numpy as np
from scipy import signal
from spectrum import aryule


def pyulear(x, order, nfft, fs):
    r"""
    From matlab:
    Power Spectral Density (PSD) estimate via Yule-Walker's method.
    Returns the PSD estimate, Pxx, of the
    discrete-time signal X.  When X is a vector, it is converted to a
    column vector and treated as a single channel.  When X is a matrix, the
    PSD is computed independently for each column and stored in the
    corresponding column of Pxx.  Pxx is the distribution of power per unit
    frequency. The frequency is expressed in units of radians/sample.
    ORDER is the order of the autoregressive (AR) model used to produce the
    PSD.

    For real signals, PYULEAR returns the one-sided PSD by default; for 
    complex signals, it returns the two-sided PSD.  Note that a one-sided 
    PSD contains the total power of the input signal.
    """
    # From matlab
    # no exemplo, nfft = 2000 (igual a num_seg), fs = 20
    [pxx, freq] = arspectra(x, order, nfft, fs)
    return pxx, freq


def arspectra(x, order, nfft, fs):
    r"""
    From matlab:
    Power Spectral Density estimate via a specified parametric method.
    Returns the order P parametric PSD estimate
    of the columns of X in the columns of Pxx.  It uses the method specified
    in METHOD.  METHOD can be one of: 'arcov', 'arburg', 'armcov' and
    'aryule'.  If X is a vector, it will be converted to a column vector
    before processing.

    For real signals, ARSPECTRA returns the one-sided PSD by default; for 
    complex signals, it returns the two-sided PSD.  Note that a one-sided PSD
    contains the total power of the input signal.
    """
    [a, p, k] = aryule(x, order)
    a = np.append(np.array([1]), a)
    [w, h] = signal.freqz(1, a, int(nfft), whole=True, fs=fs)
    sxx = np.abs(h)**2 * p

    [pxx, w] = computepsd(sxx, w, int(nfft), fs)

    return pxx, w


def computepsd(sxx, w, nfft, fs):
    r"""
    From matlab:
    Compute the one-sided or two-sided PSD or Mean-Square.

    Inputs:
    Sxx   - Whole power spectrum [Power]; it can be a vector or a matrix.
            For matrices the operation is applied to each column.
    W     - Frequency vector in rad/sample or in Hz.
    RANGE - Determines if a 'onesided' or a 'twosided' Pxx and Sxx are
            returned.
    NFFT  - Number of frequency points.
    Fs    - Sampling Frequency.
    """
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