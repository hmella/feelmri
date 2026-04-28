"""
Complex noise generation utilities for MR image simulation.
"""
import numpy as np


def add_cpx_noise(image, mask=1, std=[], relative_std=[], SNR=20, ref=0, recover_noise=False):
    """Add complex Gaussian noise to an input image.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    mask : np.ndarray or int, optional
        Mask array to apply noise selectively. Default is 1 (full image).
    std : float or np.ndarray, optional
        Standard deviation of the noise. Default is an empty list.
    relative_std : float or np.ndarray, optional
        Relative standard deviation of the noise (fraction of peak magnitude).
        Default is an empty list.
    SNR : float, optional
        Signal-to-noise ratio. Default is 20.
    ref : int, optional
        Reference index for noise calculation. Default is 0.
    recover_noise : bool, optional
        If True, also return the noise array. Default is False.

    Returns
    -------
    np.ndarray
        Image with added complex noise. If ``recover_noise`` is True, returns
        a tuple ``(noisy_image, noise)``.
    """
    # Standard deviation
    if not relative_std:
        sigma = std
    else:
        peak = np.abs(image[..., 0]).max()
        sigma = relative_std * peak

    # Noise generation and addition
    noise = np.random.normal(0, sigma, image.shape) + 1j * np.random.normal(0, sigma, image.shape)
    image_n = image + noise * mask

    if recover_noise:
        return image_n, noise
    else:
        return image_n
