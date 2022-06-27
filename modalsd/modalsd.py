import numpy as np
from scipy.linalg import hankel

def modalsd(frf, f, fs, max_modes=22, finish="plot"):

    opts = {
        "fr": [0, 2.5], # Frequency range
        "sc": [0.01, 0.05], # Frequency and damping criteria.
        "fm": "lsce" # Fit method
    }

    # Maximum modes
    opts["mm"] = max_modes

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
        # Check for mode stability
        # TODO: DISCREPÂNCIA DE RESULTADOS NA LINHA 6
        if iMode > 1:
            [mode_stab_fn[0:(iMode - 1),iMode-2], mode_stab_dr[0:(iMode - 1),iMode-2]] = \
            compare_modes(fn[iMode-1], fn[iMode-2], dr[iMode-1], dr[iMode-2], opts)
            fn_out[iMode-2, 0:(iMode-1)] = np.transpose(fn[iMode-2])
            # Remove frequencies from fnout corresponding to modes that are not
            # stable in frequency
            cond = np.logical_not(mode_stab_fn[0:(iMode - 1),iMode-2])
            # fn_out[iMode-2, cond, iMode-2] = np.nan
            fn_out[iMode-2, 0:(iMode-1)][cond] = np.nan


    if finish == "plot":
        plot_s_diagram(frf, f, mode_fn, mode_stab_fn, mode_stab_dr, opts)
    elif finish == "return":
        return frf, f, mode_fn, mode_stab_fn, mode_stab_dr
    else:
        A = np.zeros((len(dr), len(dr)))
        for i in range(0, len(dr)):
            if type(dr[i]) == int:
                A[i, 0] = dr[i]
            else:
                for j in range(0, len(dr[i])):
                    A[i,j] = dr[i][j]
        dr = A

        # A = np.array([])
        # for i in range(0, len(fn)):
        #     for j in range(0, len(fn[i])):
        #         A[i,j] = fn[i][j]
        # fn = A

        return fn_out, dr


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


def compare_modes(fn1, fn0, dr1, dr0, opts):
    """
    % Compare the locations of the natural frequencies and damping ratios
    % between two model orders. fn1 and dr1 represent the larger model order.
    % modeStabfn and modeStabdr are logical arrays of the same size as fn0 and
    % dr0, and return true if there is a frequency or damping ratio in fn1 and
    % dr1 that is within one percent of fn0 and dr0.
    """

    mode_stab_fn = np.zeros(len(fn0))
    mode_stab_dr = np.zeros(len(dr0))

    for i in range(len(fn0)):
        mode_stab_fn[i] = min(abs(fn0[i] - fn1[:])) < opts["sc"][0]*fn0[i]
        mode_stab_dr[i] = min(abs(dr0[i] - dr1[:])) < opts["sc"][1]*dr0[i]
    
    return mode_stab_fn, mode_stab_dr


def plot_s_diagram(frf, f, mode_fn, mode_stab_fn, mode_stab_dr, opts):
    f_idx = []
    first_condition = f >= opts["fr"][0]
    second_condition = f <= opts["fr"][1]
    f_idx = np.logical_and(first_condition, second_condition)

    f = f[f_idx]
    frf = frf[f_idx]

    mode_stab_fn = np.array(mode_stab_fn, dtype=bool)
    mode_stab_dr = np.array(mode_stab_dr, dtype=bool)

    i_f = np.logical_and(mode_stab_fn, np.logical_not(mode_stab_dr))
    i_f_and_d = np.logical_and(mode_stab_fn, mode_stab_dr)
    i_not_f = np.logical_not(mode_stab_fn)

    mode_number = range(1, opts["mm"]+1) * np.ones((opts["mm"], 1))

    # Compute a modal peaks function from FRFs
    mpf = np.abs(np.power(frf, 2))

    # Plot configurations
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Frequency [Hz]")
    # Modes axis
    ax1.scatter(mode_fn[i_f], mode_number[i_f], marker="o")
    ax1.scatter(mode_fn[i_f_and_d], mode_number[i_f_and_d], marker="+")
    ax1.scatter(mode_fn[i_not_f], mode_number[i_not_f], marker=",")
    ax1.set_ylim([0, opts["mm"]+0.5])
    ax1.set_xlim([f[0], f[-1]])

    # FRF axis
    ax2 = ax1.twinx()
    
    color = "tab:orange"
    ax2.set_ylabel("Magnitude", color=color)
    ax2.plot(f, mpf)
    ax2.set_yscale("log")

    plt.show()

