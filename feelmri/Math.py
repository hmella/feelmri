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

def divide_no_nan(x, y, z):
  # Return x divided by y. Return x/z in places where yi equals 0
  return x/np.array([y[i] if y[i]!=0 else z[i] for i in range(len(y))])