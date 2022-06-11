import numpy as np
from scipy.linalg import hankel

def modalsd(frf, f, fs):

    opts = {
        "fr": [0, 2.5], # Frequency range
        "sc": [0.01, 0.05], # Frequency and damping criteria.
        "mm": 22, # Maximum modes
        "fm": "lsce" # Fit method
    }

    fn = np.zeros(opts["mm"], dtype=object)
    dr = np.zeros(opts["mm"], dtype=object)
    mode_fn = np.empty((opts["mm"], opts["mm"]))
    mode_fn[:] = np.nan

    mode_stab_fn = np.zeros((opts["mm"], opts["mm"]))
    mode_stab_dr = np.zeros((opts["mm"], opts["mm"]))

    fn_out = np.zeros((opts["mm"], opts["mm"]))

    for iMode in range(1,opts["mm"]):
        opts["nx"] = 2 * iMode
        opts["ft"] = False
        [fn_i, dr_i] = poles_to_fd(compute_poles(frf, f, fs, iMode, opts))
        # Save modes

        sort_index = np.argsort(fn_i)
        fn_i = fn_i[sort_index]
        dr_i = dr_i[sort_index]
        fn[iMode-1] = fn_i
        dr[iMode-1] = dr_i
        mode_fn[0:iMode, iMode-1] = fn[iMode-1]
        if iMode > 1:
            pass
        



def compute_poles(frf, f: np.array, fs, mnum, opts):
    if (opts["fm"] == "lsce"):
        # Filtra os valores de acordo com o limite informado
        f_idx = []
        first_condition = f >= opts["fr"][0]
        second_condition = f <= opts["fr"][1]
        f_idx = np.logical_and(first_condition, second_condition)
        
        if (f_idx.any()):
            frf = frf[f_idx]
            f = f[f_idx]
            fs = round(fs*sum(f_idx)/len(f_idx), 4)
        
        L = 2 * mnum # Polynomial order
        o_factor = 10 # Number of times samples = oFactor*L
        n_out = 1 # Number of outputs
        n_in = 1 # Number of inputs

        H = np.zeros((L*o_factor*n_out*n_in, L))
        b = np.zeros((L*o_factor*n_out*n_in, 1))

        # Form the least-squares system
        h = compute_ir(frf, f, fs)
        H0 = hankel(h[0:L*o_factor], h[L*o_factor-1:L*o_factor+L])
    
        H[:,:] = H0[:, 0:-1]
        b = -1*H0[:, -1]

        # TODO: Verificar se funciona dessa forma para o numpy
        beta = np.real(np.linalg.lstsq(H, b, rcond=None)[0])
        beta = np.append(beta, 1)

        # Find the roots of the polynomial
        V = np.roots(np.flipud(beta))
        p = np.log(V)*fs

        # Remove poles that are not in complex-conjugate pairs, have positive
        # real part, or are real. For complex-conjugate pairs, keep only the
        # pole that has positive imaginary part.
        cond1 = np.imag(p) < 0
        cond2 = np.real(p) <= 0
        cond3 = [not np.isreal(p_i) for p_i in p]

        cond = np.logical_and(cond1, cond2)
        cond = np.logical_and(cond, cond3)

        p_conj = np.conj(p[cond])

        p = np.intersect1d(p, p_conj)

        # Place NaN's for excluded poles. Poles should not return empty.
        pad = np.empty(mnum - len(p))
        pad[:] = np.nan
        poles = np.append(p, pad)

        if (f_idx.any()):
            # Compensate poles for frequency offset
            [fn, dr] = poles_to_fd(poles)
            fn_ef = fn + f[0]
            dr_ef = dr * fn / (fn_ef)
            poles = fd_to_poles(fn_ef, dr_ef)

        return poles



def compute_ir(frf, f, fs):
    N = round(fs/(f[2] - f[1]))

    if (np.mod(N, 2)):
        addition = np.flipud(np.conj(frf[1:]))
        frf = np.append(frf, addition)
        h = np.real(np.fft.ifft(frf))
    else:
        addition = np.flipud(np.conj(frf[1:-1]))
        frf = frf.append(addition)
        h = np.real(np.ifft(frf))
    
    return h


def poles_to_fd(poles):
    """
    Compute natural frequency and damping from poles.
    """

    # Compute natural frequency and damping for each pole
    wn = np.abs(poles)
    fn = wn / (2*np.pi)
    dr = -1*np.real(poles) / wn
    return fn, dr


def fd_to_poles(fn, dr):
    """
    Compute poles from natural frequency and damping.
    """

    # Compute poles for each natural frequency and damping
    wn = 2 * np.pi * fn
    poles = -1*dr*wn + 1j*wn*np.sqrt(1 - dr**2)
    return poles