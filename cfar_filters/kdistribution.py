import numpy as np
from scipy.special import gamma, kn
import scipy.integrate as integrate
from scipy.optimize import fmin

# implementation of the K-distribution pdf
# E.q. 8.21 in Oliver/Quegan:
# C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.
def k_pdf(x, μ, v, L):

    v = np.float64(v)
    x = np.float64(x)
    n = np.round(v - L)  # order of bessel function must be integer

    a = 2 / (gamma(L) * gamma(v))
    b = (L * v / μ)**((L + v) / 2)
    c = x**((L + v - 2) / 2)
    d = kn(n, 2 * np.sqrt(L * v * x / μ))

    pdf = a * b * c * d

    pdf = np.where(np.isnan(pdf), 0, pdf)

    return pdf

# Numerical integration of the k-distribution using scipy.integrate
# I cannot find a nice expression for the CDF, so i'm using the pdf instead.
def _k_minimize(t, μ, v, L, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(L)))[0] - pde)

# K-distribution CFAR on image blocks
def _kd_cfar(image, μ, v, L, pde):
    init = 5 * μ  # initial guess of the algorithm
    T = fmin(_k_minimize, init, disp=False, args=(μ, v, L, pde))[0]
    outliers = image > T
    return outliers

def detector(image, mask=0, N=200, pfa=1e-12, L=10):
    """
    CFAR filter implementation based on the K-normal distribution.

    Estimation of the K-PDF parameters is based on a known L-value
    (9-11 for Sentinel-1 EW)
    (15 for ALOS-2 Wide Beam)

    The shape-parameter v is estimated using a MoM estimator mentioned in this paper:
    C. Wesche and W. Dierking,
    “Iceberg signatures and detection in SAR images in two test regions of the Weddell Sea, Antarctica,”
    J. Glaciol., vol. 58, no. 208, pp. 325-339, 2012, doi: 10.3189/2012J0G11J020.

    The filter estimates the K-distribution parameters on NxN blocks of the image to improve execution speed.
    This approach was suggested here:
    C. Liu, “Method for Fitting K-Distributed Probability Density Function to Ocean Pixels in Dual-Polarization SAR,”
    Can. J. Remote Sens., vol. 44, no. 4, pp. 299-310, 2018, doi: 10.1080/07038992.2018.1491789.

    Parameters:
    ----------
    image : numpy.ndarray (X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    N : Integer
        Block size for the estimation (tile size in the paper)
    pfa : float
        Probability of false alarm. Should be somewhere between 0-1
    L : float
        Equavalent number of looks for the SAR image (normally 9-11 for Sentinel-1 EW)
    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
        Binary outlier classification
    """

    # check if shapes are correct
    if len(image.shape) != 2:
        raise ValueError(f'Input image must be of shape [X, Y] but is of shape {image.shape}')
    if (not isinstance(mask, np.ndarray)) | (mask.dtype != np.bool):
        raise TypeError(f'Input mask must be of type np.ndarray(bool) but is of type {type(mask)}, {mask.dtype}')
    if image.shape != mask.shape:
        raise ValueError((f'Shape of mask must match shape of image. \
                          Mask shape: {mask.shape}. Image shape {image.shape}'))

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    # apply mask
    image = np.where(mask, image, np.nan)

    req_valid_samples = 1000  # minimum number of valid samples in block

    outliers = np.zeros_like(image).astype(np.bool)
    pde = 1 - (pfa)  # probability of true detection

    n_rows, n_cols = np.asarray(image.shape[0:2]) // N + (np.mod(np.asarray(image.shape[0:2]), N) > 0) * 1

    # loop over blocks (tiles)
    for x in range(0, n_rows):
        for y in range(0, n_cols):

            sub_block_image = image[x * N:x * N + N, y * N:y * N + N]

            # count number of pixels not 0 and nan
            no_valid_samples = np.sum(~np.isnan(sub_block_image) & (sub_block_image != 0))

            # if block is masked then skip the block
            if no_valid_samples <= req_valid_samples:
                outliers[x * N:x * N + N, y * N:y * N + N] = np.zeros_like(sub_block_image) > 0
            else:
                # MoM estimation of the v-parameter
                μ = np.nanmean(sub_block_image)
                v = μ**2 * (L + 1) / (np.var(sub_block_image) * L - μ**2)

                outliers[x * N:x * N + N, y * N:y * N + N] = _kd_cfar(sub_block_image, μ, v, L, pde)

    outliers = np.where(mask, outliers, False)
    return outliers
