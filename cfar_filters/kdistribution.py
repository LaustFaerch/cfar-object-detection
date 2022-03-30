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
def _k_minimize(t, μ, v, L, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(L)))[0] - pde)

# The full MoM estimation suggested by C. Liu
def _mom_estimation_full(image):

    median = np.nanmedian(image)
    image = np.where(image > 3 * median, np.nan, image)

    μ = np.nanmean(image)

    I2 = np.nanmean((image)**2)  # 2nd raw sample moment
    I3 = np.nanmean((image)**3)  # 3rd raw sample moment

    W2 = I2 / μ**2  # e.q. 11
    W3 = (I3 / μ**3) / (W2)  # e.q. 11

    b = (W3 - 4 * W2 - 1)
    D = b**2 - 16 * W2

    # If the discriminant is below zero, we have no solution
    if D <= 0:
        return μ, np.nan, np.nan

    # e.q. 12
    Vplus = 1 / 4 * (-b + np.sqrt(D))
    Vminus = 1 / 4 * (-b - np.sqrt(D))
    V = max(Vplus, Vminus)
    U = W2 / V

    # e.q. 10
    L = 1 / (U - 1)
    v = 1 / (V - 1)

    return μ, v, L

# Simple MoM estimation using the system ENL
# V is then estimated based on e.q. 6 in the paper
def _mom_estimation_simple(image, L):

    μ = np.nanmean(image)
    K = np.nanmean(image**2) / μ**2
    v = (L + 1) / (L * K - L - 1)

    return μ, v, L

# K-distribution CFAR on image blocks
def _kd_cfar(image, μ, v, L, pde):
    init = 5 * μ  # initial guess of the algorithm
    T = fmin(_k_minimize, init, disp=False, args=(μ, v, L, pde))[0]
    outliers = image > T
    return outliers


def detector(image, mask=0, N=200, pfa=1e-12, offset=False, enl=10):
    """
    CFAR filter implementation based on the K-normal distribution.
    The filter is based on the paper:
    C. Liu, “Method for Fitting K-Distributed Probability Density Function to Ocean Pixels in Dual-Polarization SAR,”
    Can. J. Remote Sens., vol. 44, no. 4, pp. 299-310, 2018, doi: 10.1080/07038992.2018.1491789.

    The filter estimated the K-distribution parameters on NxN blocks of the image.
    Parameters are estimated using the MoM method suggested in the paper


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
    offset: np.bool
        Flag used to select whether the sub blocks should be offset to 0
        The MoM estimation works best of the distribution starts near 0
        Set offset = True for the dpolrad transform
        Set Offset = False for the NIS transform

    enl : float
        Equavalent number of looks for the SAR image (normally 9-11 for Sentinel-1 EW)
        If using a linear transformation image (e.g. DPolRad/NIS), then remember that the number of looks are influenced
        From C. Liu: "When a combined decision variable is derived from Equation (1),
        the maximum possible value of L is the sum of the number of looks of the channels."

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

    vmin, vmax = 1, 50
    Lmin, Lmax = 1, enl
    req_valid_samples = 2500  # minimum number of valid samples in block

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

                if offset:
                    # offset to ensure pdf starts near 0 (which is important for the mom estimation)
                    # use offset for dpolrad transform, NOT for the chenliu transform
                    sub_block_image = sub_block_image - np.nanmin(sub_block_image)

                # ESTIMATE PARAMETERS FOR THE K-DISTRIBUTION
                μ, v, L = _mom_estimation_full(sub_block_image)

                # if the mom estimation above fails, use a simpler estimation based on the system ENL
                if np.any(np.isnan(np.array([v, L]))):
                    μ, v, L = _mom_estimation_simple(sub_block_image, enl)

                # If v is negative it is likely that the equations broke down
                # due to the denominator in the simple mom estimation
                # If v is very large, the distribution is likely near gamma
                # In both cases use vmax.
                if v <= vmin or v >= vmax:
                    v = vmax
                # L cannot be negative
                if L <= 0:
                    L = Lmax
                # L cannot be smaller than 1 or larger than p*ENL
                L = min(max(Lmin, L), Lmax)

                outliers[x * N:x * N + N, y * N:y * N + N] = _kd_cfar(sub_block_image, μ, v, L, pde)

    outliers = np.where(mask, outliers, False)
    return outliers
