"""
Math utility functions: k-space/image-space Fourier transforms and 3D rotation matrices.
"""
import numpy as np
from numpy.fft import fftn, fftshift, ifftn, ifftshift


def itok(x, axes=None):
    """N-dimensional Fourier transform from image space to k-space.

    Parameters
    ----------
    x : np.ndarray
        Input image array.
    axes : list of int, optional
        Axes along which to apply the transform. Defaults to the first
        three spatial dimensions.

    Returns
    -------
    np.ndarray
        k-space representation of the input.
    """
    if axes is None:
        axes = [i for i in range(len(x.shape)) if i < 3]
    return fftshift(fftn(ifftshift(x, axes=axes), axes=axes), axes=axes)


def ktoi(x, axes=None):
    """N-dimensional inverse Fourier transform from k-space to image space.

    Parameters
    ----------
    x : np.ndarray
        Input k-space array.
    axes : list of int, optional
        Axes along which to apply the transform. Defaults to the first
        three spatial dimensions.

    Returns
    -------
    np.ndarray
        Image-space representation of the input.
    """
    if axes is None:
        axes = [i for i in range(len(x.shape)) if i < 3]
    return fftshift(ifftn(ifftshift(x, axes=axes), axes=axes), axes=axes)


def Rx(tx):
    """Rotation matrix around the X-axis.

    Parameters
    ----------
    tx : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3×3 rotation matrix.
    """
    return np.array([
        [1,           0,            0],
        [0,  np.cos(tx), -np.sin(tx)],
        [0,  np.sin(tx),  np.cos(tx)],
    ])


def Ry(ty):
    """Rotation matrix around the Y-axis.

    Parameters
    ----------
    ty : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3×3 rotation matrix.
    """
    return np.array([
        [ np.cos(ty), 0, np.sin(ty)],
        [           0, 1,          0],
        [-np.sin(ty), 0, np.cos(ty)],
    ])


def Rz(tz):
    """Rotation matrix around the Z-axis.

    Parameters
    ----------
    tz : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3×3 rotation matrix.
    """
    return np.array([
        [np.cos(tz), -np.sin(tz), 0],
        [np.sin(tz),  np.cos(tz), 0],
        [          0,           0, 1],
    ])
