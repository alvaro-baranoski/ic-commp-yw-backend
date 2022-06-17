import numpy as np

def aryule(X, order, norm='biased', allow_singularity=True):
    r"""Compute AR coefficients using Yule-Walker method
    :param X: Array of complex data values, X(1) to X(N)
    :param int order: Order of autoregressive process to be fitted (integer)
    :param str norm: Use a biased or unbiased correlation.
    :param bool allow_singularity:
    :return:
        * AR coefficients (complex)
        * variance of white noise (Real)
        * reflection coefficients for use in lattice filter
    .. rubric:: Description:
    The Yule-Walker method returns the polynomial A corresponding to the
    AR parametric signal model estimate of vector X using the Yule-Walker
    (autocorrelation) method. The autocorrelation may be computed using a
    **biased** or **unbiased** estimation. In practice, the biased estimate of
    the autocorrelation is used for the unknown true autocorrelation. Indeed,
    an unbiased estimate may result in nonpositive-definite autocorrelation
    matrix.
    So, a biased estimate leads to a stable AR filter.
    The following matrix form represents the Yule-Walker equations. The are
    solved by means of the Levinson-Durbin recursion:
     .. math::
        \left( \begin{array}{cccc}
        r(1) & r(2)^* & \dots & r(n)^*\\
        r(2) & r(1)^* & \dots & r(n-1)^*\\
        \dots & \dots & \dots & \dots\\
        r(n) & \dots & r(2) & r(1) \end{array} \right)
        \left( \begin{array}{cccc}
        a(2)\\
        a(3) \\
        \dots \\
        a(n+1)  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r(2)\\
        -r(3) \\
        \dots \\
        -r(n+1)  \end{array} \right)
    The outputs consists of the AR coefficients, the estimated variance of the
    white noise process, and the reflection coefficients. These outputs can be
    used to estimate the optimal order by using :mod:`~spectrum.criteria`.
    .. rubric:: Examples:
    From a known AR process or order 4, we estimate those AR parameters using
    the aryule function.
    .. doctest::
        >>> from scipy.signal import lfilter
        >>> from spectrum import *
        >>> from numpy.random import randn
        >>> A  =[1, -2.7607, 3.8106, -2.6535, 0.9238]
        >>> noise = randn(1, 1024)
        >>> y = lfilter([1], A, noise);
        >>> #filter a white noise input to create AR(4) process
        >>> [ar, var, reflec] = aryule(y[0], 4)
        >>> # ar should contains values similar to A
    The PSD estimate of a data samples is computed and plotted as follows:
    .. plot::
        :width: 80%
        :include-source:
        from spectrum import *
        from pylab import *
        ar, P, k = aryule(marple_data, 15, norm='biased')
        psd = arma2psd(ar)
        plot(linspace(-0.5, 0.5, 4096), 10 * log10(psd/max(psd)))
        axis([-0.5, 0.5, -60, 0])
    .. note:: The outputs have been double checked against (1) octave outputs
        (octave has norm='biased' by default) and (2) Marple test code.
    .. seealso:: This function uses :func:`~spectrum.levinson.LEVINSON` and
        :func:`~spectrum.correlation.CORRELATION`. See the :mod:`~spectrum.criteria`
        module for criteria to automatically select the AR order.
    :References: [Marple]_
    """
    assert norm in ['biased', 'unbiased']
    r = CORRELATION(X, maxlags=order, norm=norm)
    A, P, k = LEVINSON(r, allow_singularity=allow_singularity)
    return A, P, k

def CORRELATION(x, y=None, maxlags=None, norm='unbiased'):
    r"""Correlation function
    This function should give the same results as :func:`xcorr` but it
    returns the positive lags only. Moreover the algorithm does not use
    FFT as compared to other algorithms.
    :param array x: first data array of length N
    :param array y: second data array of length N. If not specified, computes the
        autocorrelation.
    :param int maxlags: compute cross correlation between [0:maxlags]
        when maxlags is not specified, the range of lags is [0:maxlags].
    :param str norm: normalisation in ['biased', 'unbiased', None, 'coeff']
        * *biased*   correlation=raw/N,
        * *unbiased* correlation=raw/(N-`|lag|`)
        * *coeff*    correlation=raw/(rms(x).rms(y))/N
        * None       correlation=raw
    :return:
        * a numpy.array correlation sequence,  r[1,N]
        * a float for the zero-lag correlation,  r[0]
    The *unbiased* correlation has the form:
    .. math::
        \hat{r}_{xx} = \frac{1}{N-m}T \sum_{n=0}^{N-m-1} x[n+m]x^*[n] T
    The *biased* correlation differs by the front factor only:
    .. math::
        \check{r}_{xx} = \frac{1}{N}T \sum_{n=0}^{N-m-1} x[n+m]x^*[n] T
    with :math:`0\leq m\leq N-1`.
    .. doctest::
        >>> from spectrum import CORRELATION
        >>> x = [1,2,3,4,5]
        >>> res = CORRELATION(x,x, maxlags=0, norm='biased')
        >>> res[0]
        11.0
    .. note:: this function should be replaced by :func:`xcorr`.
    .. seealso:: :func:`xcorr`
    """
    assert norm in ['unbiased','biased', 'coeff', None]
    #transform lag into list if it is an integer
    x = np.array(x)
    if y is None:
        y = x
    else:
        y = np.array(y)

    # N is the max of x and y
    N = max(len(x), len(y))
    if len(x) < N:
        x = y.copy()
        x.resize(N, refcheck=False)
    if len(y) < N:
        y = y.copy()
        y.resize(N, refcheck=False)

    #default lag is N-1
    if maxlags is None:
        maxlags = N - 1
    assert maxlags < N, 'lag must be less than len(x)'

    realdata = np.isrealobj(x) and np.isrealobj(y)
    #create an autocorrelation array with same length as lag
    if realdata == True:
        r = np.zeros(maxlags, dtype=float)
    else:
        r = np.zeros(maxlags, dtype=complex)

    if norm == 'coeff':
        rmsx = pylab_rms_flat(x)
        rmsy = pylab_rms_flat(y)

    for k in range(0, maxlags+1):
        nk = N - k - 1

        if realdata == True:
            sum = 0
            for j in range(0, nk+1):
                sum = sum + x[j+k] * y[j]
        else:
            sum = 0. + 0j
            for j in range(0, nk+1):
                sum = sum + x[j+k] * y[j].conjugate()
        if k == 0:
            if norm in ['biased', 'unbiased']:
                r0 = sum/float(N)
            elif norm is None:
                r0 = sum
            else:
                r0 =  1.
        else:
            if norm == 'unbiased':
                r[k-1] = sum / float(N-k)
            elif norm == 'biased':
                r[k-1] = sum / float(N)
            elif norm is None:
                r[k-1] = sum
            elif norm == 'coeff':
                r[k-1] =  sum/(rmsx*rmsy)/float(N)

    r = np.insert(r, 0, r0)
    return r

def pylab_rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    (Copied 1:1 from matplotlib.mlab.)
    """
    return np.sqrt(np.mean(np.absolute(a) ** 2))


def LEVINSON(r, order=None, allow_singularity=False):
    r"""Levinson-Durbin recursion.
    Find the coefficients of a length(r)-1 order autoregressive linear process
    :param r: autocorrelation sequence of length N + 1 (first element being the zero-lag autocorrelation)
    :param order: requested order of the autoregressive coefficients. default is N.
    :param allow_singularity: false by default. Other implementations may be True (e.g., octave)
    :return:
        * the `N+1` autoregressive coefficients :math:`A=(1, a_1...a_N)`
        * the prediction errors
        * the `N` reflections coefficients values
    This algorithm solves the set of complex linear simultaneous equations
    using Levinson algorithm.
    .. math::
        \bold{T}_M \left( \begin{array}{c} 1 \\ \bold{a}_M \end{array} \right) =
        \left( \begin{array}{c} \rho_M \\ \bold{0}_M  \end{array} \right)
    where :math:`\bold{T}_M` is a Hermitian Toeplitz matrix with elements
    :math:`T_0, T_1, \dots ,T_M`.
    .. note:: Solving this equations by Gaussian elimination would
        require :math:`M^3` operations whereas the levinson algorithm
        requires :math:`M^2+M` additions and :math:`M^2+M` multiplications.
    This is equivalent to solve the following symmetric Toeplitz system of
    linear equations
    .. math::
        \left( \begin{array}{cccc}
        r_1 & r_2^* & \dots & r_{n}^*\\
        r_2 & r_1^* & \dots & r_{n-1}^*\\
        \dots & \dots & \dots & \dots\\
        r_n & \dots & r_2 & r_1 \end{array} \right)
        \left( \begin{array}{cccc}
        a_2\\
        a_3 \\
        \dots \\
        a_{N+1}  \end{array} \right)
        =
        \left( \begin{array}{cccc}
        -r_2\\
        -r_3 \\
        \dots \\
        -r_{N+1}  \end{array} \right)
    where :math:`r = (r_1  ... r_{N+1})` is the input autocorrelation vector, and
    :math:`r_i^*` denotes the complex conjugate of :math:`r_i`. The input r is typically
    a vector of autocorrelation coefficients where lag 0 is the first
    element :math:`r_1`.
    .. doctest::
        >>> import numpy; from spectrum import LEVINSON
        >>> T = numpy.array([3., -2+0.5j, .7-1j])
        >>> a, e, k = LEVINSON(T)
    """
    #from numpy import isrealobj
    T0  = np.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = np.isrealobj(r)
    if realdata is True:
        A = np.zeros(M, dtype=float)
        ref = np.zeros(M, dtype=float)
    else:
        A = np.zeros(M, dtype=complex)
        ref = np.zeros(M, dtype=complex)

    P = T0

    for k in range(0, M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            #save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k-j-1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp**2.)
        else:
            P = P * (1. - (temp.real**2+temp.imag**2))
        if P <= 0 and allow_singularity==False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k+1)//2
        if realdata is True:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp*save
        else:
            for j in range(0, khalf):
                kj = k-j-1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref