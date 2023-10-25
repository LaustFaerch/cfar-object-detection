import warnings
import numpy as np
from scipy.special import gammaincc
from scipy.optimize import minimize
from .utils import smells_like, mask_edges
from .fast_functions import fast_edge_mean


def _gamma_pfa(t, L):
    # scipy.stats.gammaincc is already regualized with 1/gamma(L)
    return gammaincc(L, t * L)

"""
Solves E.q. 17 in D.J. Crisp, E.q., 10.14 in Oliver/Quegan
"""
def _gamma_pfa_minimization(x, pfa, L):
    return np.abs(_gamma_pfa(x, L) - pfa)

# Find the required multiplier for a desired PFA level numerically
def _find_gamma_multiplier(pfa, L):
    x0 = 3  # initial guess
    res = minimize(_gamma_pfa_minimization, x0, args=(pfa, L), method='Nelder-Mead', tol=1e-6)

    # TODO: I have has some convergence issues.
    # Maybe test if using another method than 'Nelder-Mead' gives better results
    # For now, I have just added a warning
    if res.x[0] == x0:
        warnings.warn('gamma CFAR might not have converged', category=UserWarning)

    return res.x[0]


def detector(image, mask=0, pfa=1e-12, enl=10, wi=9, wo=15):
    """
    CFAR filter implementation based on the gamma distribution.
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
    if smells_like(image) != 'intensity':
        warnings.warn(f'Input image should be in intensity scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    multiplier = _find_gamma_multiplier(pfa, enl)

    edge_mean = fast_edge_mean(image, mask, wi, wo)

    Δ = (image > (edge_mean * multiplier))

    outliers = mask_edges(Δ, 20, False)

    return outliers
