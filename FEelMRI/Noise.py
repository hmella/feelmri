import numpy as np


# Get tissue and background indicators
def get_segmentation_mask(u_image, tol=1.0e-15):
    '''
    '''
    # Background pixels
    condition = np.abs(u_image) < tol
    tissue      = np.where(~condition)
    background  = np.where(condition)

#    # Get np.pixels with tissue
#    condition = u_image != 0.0
#    tissue      = np.where(condition)
#    background  = np.where(~condition)
#
    return tissue, background


# Add complex noise
def add_cpx_noise(image, mask=[], std=[], relative_std=[], SNR=20, ref=0, recover_noise=False):
  """
  Add complex noise to an input image.

  Parameters:
  image (ndarray): Input image array.
  mask (ndarray, optional): Mask array to apply noise selectively. Default is an empty list.
  std (float or ndarray, optional): Standard deviation of the noise. Default is an empty list.
  relative_std (float or ndarray, optional): Relative standard deviation of the noise. Default is an empty list.
  SNR (float, optional): Signal-to-noise ratio. Default is 20.
  ref (int, optional): Reference index for noise calculation. Default is 0.
  recover_noise (bool, optional): If True, return the noise added. Default is False.

  Returns:
  ndarray: Image with added noise.
  tuple: (Image with added noise, noise) if recover_noise is True.
  """

  # Standard deviation
  if not relative_std:
    sigma = std
  else:  	   
    peak = np.abs(image[...,0]).max()
    sigma = relative_std*peak

  # Noise generation and addition
  noise = np.random.normal(0, sigma, image.shape) + 1j*np.random.normal(0, sigma, image.shape)
  image_n = image + noise*mask

  if recover_noise:
    return image_n, noise
  else:
    return image_n
