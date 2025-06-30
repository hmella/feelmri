import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift

MEGA = 1e+06
KILO = 1e+03
MILI = 1e-02
MICRO = 1e-06


def itok(x, axes=None):
  # N-dimensional Fast Fourier Transform (image to k-space) through dimensions given in axes
  if axes is None:
    axes = [i for i in range(len(x.shape)) if i < 3]
  return fftshift(fftn(ifftshift(x, axes=axes), axes=axes), axes=axes)

def ktoi(x, axes=None):
  # N-dimensional inverse Fast Fourier Transform (k-space to image) through dimensions given in axes
  if axes is None:
    axes = [i for i in range(len(x.shape)) if i < 3]
  return fftshift(ifftn(ifftshift(x, axes=axes), axes=axes), axes=axes)

def wrap(x, value):
  # Wrap-to-value function
  return np.mod(x + 0.5*value, value) - 0.5*value

def Rx(tx):
  # Rotation matrix around X-axis for a rotation angle of tx (in radians)
  return np.array([[1, 0, 0],
                    [0, np.cos(tx), -np.sin(tx)],
                    [0, np.sin(tx), np.cos(tx)]])

def Ry(ty):
  # Rotation matrix around Y-axis for a rotation angle of tx (in radians)
  return np.array([[np.cos(ty), 0, np.sin(ty)],
                    [0, 1, 0],
                    [-np.sin(ty), 0, np.cos(ty)]])

def Rz(tz):
  # Rotation matrix around Z-axis for a rotation angle of tx (in radians)
  return np.array([[np.cos(tz), -np.sin(tz), 0],
                    [np.sin(tz), np.cos(tz), 0],
                    [0, 0, 1]])

def faster_polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    If `p` is of length N, this function returns the value::

        p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    If `x` is a sequence, then ``p(x)`` is returned for each element of ``x``.
    If `x` is another polynomial then the composite polynomial ``p(x(t))``
    is returned.

    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.

    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.

    If `x` is a subtype of `ndarray` the return value will be of the same type.

    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.

    Examples
    --------
    >>> import numpy as np
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([76])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([76])

    """
    y = np.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y

def divide_no_nan(x, y, z):
  # Return x divided by y. Return x/z in places where yi equals 0
  return x/np.array([y[i] if y[i]!=0 else z[i] for i in range(len(y))])