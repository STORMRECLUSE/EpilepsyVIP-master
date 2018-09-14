__author__ = 'vsp'
import numpy as np

def square_window_spectrum(N, Fs):
    r"""
    Calculate the analytical spectrum of a square window

    Parameters
    ----------
    N : int
       the size of the window

    Fs : float
       The sampling rate

    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands

    Notes
    -----
    This is equation 21c in Harris (1978):

    .. math::

      W(\theta) = exp(-j \frac{N-1}{2} \theta) \frac{sin \frac{N\theta}{2}} {sin\frac{\theta}{2}}

    F.J. Harris (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66:51-83
    """
    f = get_freqs(Fs, N - 1)
    j = 0 + 1j
    a = -j * (N - 1) * f / 2
    b = np.sin(N * f / 2.0)
    c = np.sin(f / 2.0)
    make = np.exp(a) * b / c

    return f,  make[1:] / make[1]


def hanning_window_spectrum(N, Fs):
    r"""
    Calculate the analytical spectrum of a Hanning window

    Parameters
    ----------
    N : int
       The size of the window

    Fs : float
       The sampling rate

    Returns
    -------
    float array - the frequency bands, given N and FS
    complex array: the power in the spectrum of the square window in the
    frequency bands

    Notes
    -----
    This is equation 28b in Harris (1978):

    .. math::

      W(\theta) = 0.5 D(\theta) + 0.25 (D(\theta - \frac{2\pi}{N}) +
                D(\theta + \frac{2\pi}{N}) ),

    where:

    .. math::

      D(\theta) = exp(j\frac{\theta}{2})
                  \frac{sin\frac{N\theta}{2}}{sin\frac{\theta}{2}}

    F.J. Harris (1978). On the use of windows for harmonic analysis with the
    discrete Fourier transform. Proceedings of the IEEE, 66:51-83
    """
    #A helper function
    D = lambda theta, n: (
        np.exp((0 + 1j) * theta / 2) * ((np.sin(n * theta / 2)) / (theta / 2)))

    f = get_freqs(Fs, N)

    make = 0.5 * D(f, N) + 0.25 * (D((f - (2 * np.pi / N)), N) +
                                   D((f + (2 * np.pi / N)), N))
    return f, make[1:] / make[1]


def ar_generator(N=512, sigma=1., coefs=None, drop_transients=0, v=None):
    """
    This generates a signal u(n) = a1*u(n-1) + a2*u(n-2) + ... + v(n)
    where v(n) is a stationary stochastic process with zero mean
    and variance = sigma. XXX: confusing variance notation

    Parameters
    ----------

    N : int
      sequence length
    sigma : float
      power of the white noise driving process
    coefs : sequence
      AR coefficients for k = 1, 2, ..., P
    drop_transients : int
      number of initial IIR filter transient terms to drop
    v : ndarray
      custom noise process

    Parameters
    ----------

    N : float
       The number of points in the AR process generated. Default: 512
    sigma : float
       The variance of the noise in the AR process. Default: 1
    coefs : list or array of floats
       The AR model coefficients. Default: [2.7607, -3.8106, 2.6535, -0.9238],
       which is a sequence shown to be well-estimated by an order 8 AR system.
    drop_transients : float
       How many samples to drop from the beginning of the sequence (the
       transient phases of the process), so that the process can be considered
       stationary.
    v : float array
       Optionally, input a specific sequence of noise samples (this over-rides
       the sigma parameter). Default: None

    Returns
    -------

    u : ndarray
       the AR sequence
    v : ndarray
       the unit-variance innovations sequence
    coefs : ndarray
       feedback coefficients from k=1,len(coefs)

    The form of the feedback coefficients is a little different than
    the normal linear constant-coefficient difference equation. Therefore
    the transfer function implemented in this method is

    H(z) = sigma**0.5 / ( 1 - sum_k coefs(k)z**(-k) )    1 <= k <= P

    Examples
    --------

    >>> import nitime.algorithms as alg
    >>> ar_seq, nz, alpha = ar_generator()
    >>> fgrid, hz = alg.freq_response(1.0, a=np.r_[1, -alpha])
    >>> sdf_ar = (hz * hz.conj()).real

    """
    if coefs is None:
        # this sequence is shown to be estimated well by an order 8 AR system
        coefs = np.array([2.7607, -3.8106, 2.6535, -0.9238])
    else:
        coefs = np.asarray(coefs)

    # The number of terms we generate must include the dropped transients, and
    # then at the end we cut those out of the returned array.
    N += drop_transients

    # Typically uses just pass sigma in, but optionally they can provide their
    # own noise vector, case in which we use it
    if v is None:
        v = np.random.normal(size=N)
        v -= v[drop_transients:].mean()

    b = [sigma ** 0.5]
    a = np.r_[1, -coefs]
    u = sig.lfilter(b, a, v)

    # Only return the data after the drop_transients terms
    return u[drop_transients:], v[drop_transients:], coefs


def circularize(x, bottom=0, top=2 * np.pi, deg=False):
    """Maps the input into the continuous interval (bottom, top) where
    bottom defaults to 0 and top defaults to 2*pi

    Parameters
    ----------

    x : ndarray - the input array

    bottom : float, optional (defaults to 0).
        If you want to set the bottom of the interval into which you
        modulu to something else than 0.

    top : float, optional (defaults to 2*pi).
        If you want to set the top of the interval into which you
        modulu to something else than 2*pi

    Returns
    -------
    The input array, mapped into the interval (bottom,top)

    """
    x = np.asarray([x])

    if  (np.all(x[np.isfinite(x)] >= bottom) and
         np.all(x[np.isfinite(x)] <= top)):
        return np.squeeze(x)
    else:
        x[np.where(x < 0)] += top
        x[np.where(x > top)] -= top

    return np.squeeze(circularize(x, bottom=bottom, top=top))


def dB(x, power=True):
    """Convert the values in x to decibels.
    If the values in x are in 'power'-like units, then set the power
    flag accordingly

    1) dB(x) = 10log10(x)                     (if power==True)
    2) dB(x) = 10log10(|x|^2) = 20log10(|x|)  (if power==False)
    """
    if not power:
        return 20 * np.log10(np.abs(x))
    return 10 * np.log10(np.abs(x))

