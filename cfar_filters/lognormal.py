import warnings
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from .utils import smells_like, mask_edges
from .fast_functions import fast_edge_mean, fast_edge_std


# This functions assumes standardized data
def _gaussian_pfa(t):
    return 0.5 - 0.5 * erf(t / np.sqrt(2))

def _gaussian_pfa_minimization(x, pfa):
    return np.abs(_gaussian_pfa(x) - pfa)

# Finc the required multiplier for a desired PFA level numerically
def _find_gaussian_multiplier(pfa):
    res = minimize(_gaussian_pfa_minimization, 5, args=(pfa), method='Nelder-Mead', tol=1e-6)
    return res.x[0]


def detector(image, mask=0, pfa=1e-12, wi=9, wo=15):
    """
    CFAR filter implementation based on the log-normal distribution.
    For further details, please see Section 5.2 in this report:
    D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.

    Parameters:
    ----------
    image : numpy.ndarray(float32) (X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    pfa : float
        Probability of false alarm. Should be somewhere between 0-1
    wi : integer
        Inner Window Diameter - also called guard area
    wo : integer
        Outer Window Diameter - also called clutter estimation area

    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
        Binary outlier classification
    """

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image) > 0

    # check if shapes are correct
    if len(image.shape) != 2:
        raise ValueError(f'Input image must be of shape [X, Y] but is of shape {image.shape}')
    if image.shape != mask.shape:
        raise ValueError((f'Shape of mask must match shape of image. \
                          Mask shape: {mask.shape}. Image shape {image.shape}'))
    # check if the image format
    if smells_like(image) != 'decibel':
        warnings.warn(f'Input image should be in decibel scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    image = (image - np.nanmean(image)) / np.nanstd(image)  # standardize the data

    T = _find_gaussian_multiplier(pfa)

    edge_mean = fast_edge_mean(image, mask, wi, wo)
    egde_std = fast_edge_std(image, mask, wi, wo)

    Δ = (image > (edge_mean + T * egde_std))

    outliers = mask_edges(Δ, 20, False)

    return outliers
