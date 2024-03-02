import warnings
import numpy as np
import pandas as pd

import scipy.integrate as integrate
from scipy.special import gamma, kn
from scipy.optimize import fmin

from .utils import smells_like, mask_edges
from .fast_functions import fast_edge_mean, fast_edge_std

# implementation of the K-distribution pdf
# E.q. 8.21 in Oliver/Quegan:
# C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.
def k_pdf(x, μ, v, L):

    v = np.float64(v)
    x = np.float64(x)
    n = np.round(v - L)  # order of bessel function must be integer

    a = 2 / (gamma(L) * gamma(v))
    b = (L * v / μ)**((L + v) / 2)
    c = x**((L + v - 2) / 2)  # TODO: supress RuntimeWarning: invalid value encountered in double_scalars
    d = kn(n, 2 * np.sqrt(L * v * x / μ))  # TODO: supress RuntimeWarning: invalid value encountered in sqrt

    if np.any(np.array([a, b, c, d]) == 0):
        return 0
    else:
        pdf = a * b * c * d
        pdf = np.where(np.isnan(pdf), 0, pdf)
        return pdf

# Numerical integration of the k-distribution using scipy.integrate
# I cannot find a nice expression for the CDF, so i'm using the pdf instead.
def _k_minimize(t, μ, v, L, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(L)))[0] - pde)

# Get K-CFAR threshold
def get_threshold(row, enl, pfa):
    v = row['v']
    pde = 1 - pfa
    T = np.round(fmin(_k_minimize, 5, disp=False, args=(1, v, enl, pde))[0], 2)
    return np.round(T, 2)


def detector(image, mask=0, N=40, pfa=1e-12, enl=10, wi=9, wo=15):
    """
    CFAR filter implementation based on the K-normal distribution.

    Estimation of the K-PDF parameters is based on a known enl-value
    (9-11 for Sentinel-1 EW)
    (15 for ALOS-2 Wide Beam)

    The order-parameter (v) is estimated using a MoM estimator mentioned in this paper:
    C. Wesche and W. Dierking,
    “Iceberg signatures and detection in SAR images in two test regions of the Weddell Sea, Antarctica,”
    J. Glaciol., vol. 58, no. 208, pp. 325-339, 2012, doi: 10.3189/2012J0G11J020.

    To speed up execution, a look-up table of thresholds based on min/max order parameters are calculated first.
    This is similar to the method suggested by C. Brekke:
    C. Brekke, Automatic ship detection based on satellite SAR, no. May. 2009.
    The threshold is calculated at predefined steps, which are set by the N parameter. Higher N means finer increments.
    We do not interpolate from the LUT, but uses the nearest suitable value.
    The order parameter (v) is clipped between 1-20. This is based on the observed range of the parameter but might need
    refinement for other images.

    Parameters:
    ----------
    image : numpy.ndarray (X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    N : Integer
        Number of v-estimations
    pfa : float
        Probability of false alarm. Should be somewhere between 0-1
    enl : float
        Equavalent number of looks for the SAR image (normally 9-11 for Sentinel-1 EW)
    wi : integer
        Inner Window Diameter - also called guard area
    wo : integer
        Outer Window Diameter - also called clutter estimation area

    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
        Binary outlier classification
    """

    # check if shapes are correct
    if len(image.shape) != 2:
        raise ValueError(f'Input image must be of shape [X, Y] but is of shape {image.shape}')
    if (not isinstance(mask, np.ndarray)) | (mask.dtype != bool):
        raise TypeError(f'Input mask must be of type np.ndarray(bool) but is of type {type(mask)}, {mask.dtype}')
    if image.shape != mask.shape:
        raise ValueError((f'Shape of mask must match shape of image. \
                          Mask shape: {mask.shape}. Image shape {image.shape}'))
    # check if the image format
    if smells_like(image) != 'intensity':
        warnings.warn(f'Input image should be in intensity scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    # large v means gamma distributed clutter. v cannot be negative
    vmin, vmax = 1, 20

    # calculate the LUT based on the PFA and ENL
    v_lut = pd.DataFrame(columns=['v'])
    v_lut['v'] = np.linspace(vmin, vmax, N)
    v_lut['T'] = v_lut.apply(lambda row: get_threshold(row, enl, pfa), axis=1)

    # calculate the clutter mean and variance
    edge_mean = fast_edge_mean(image, mask, wi, wo)
    egde_var = fast_edge_std(image, mask, wi, wo)**2

    # MoM estimation of the order parameter (v)
    order_param = edge_mean**2 * (enl + 1) / np.where(mask, (egde_var * enl - edge_mean**2), np.nan)
    order_param = np.where(order_param < vmin, vmin, order_param)  # clip order parameter from 1-20
    order_param = np.where(order_param > vmax, vmax, order_param)  # clip order parameter from 1-20

    # get the LUT index for fast accessing
    v_idx = np.round((order_param * vmax) / N)
    v_idx = np.where(np.isnan(v_idx), 0, v_idx).astype(np.uint)

    # get the desired threshold for each pixel in the image
    threshold = np.asarray(v_lut)[:, 1][v_idx]
    # apply the detection
    Δ = image > (threshold * edge_mean)

    outliers = mask_edges(Δ, 20, False)

    return outliers
