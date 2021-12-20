import warnings
import numpy as np
from scipy.stats import chi2
from rs_utils.sar_functions import smells_like, db2in
from .fast_functions import fast_edge_mean

# Implements equation 17+18+19 in the article
def _calc_prob(p, n, m, lnQ):
    r = 1 - ((2 * p**2 - 1) / (6 * p)) * (1 / n + 1 / m - 1 / (n + m))
    ω2 = -p**2 / 4 * (1 - 1 / r)**2 + (p**2 * (p**2 - 1) / (24)) * (
        (1 / n**2) + (1 / m**2) - (1 / (n + m)**2)) * (1 / r**2)

    z = -2 * r * lnQ
    P = (1 - ω2) * chi2.cdf(z, df=p**2) + ω2 * chi2.cdf(z, df=p**2 + 4)
    return P


def detector(image, mask=0, pfa=1e-6, enl=10):
    """
    CFAR filter implementation based on the Wishart distributions.
    Based on eq. 15 in this paper:
    K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver,
    “A test statistic in the complex wishart distribution and its application
    to change detection in polarimetric SAR data,”
    IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4-19, 2003, doi: 10.1109/TGRS.2002.808066.
    At the moment, the filter is only suitable for Sentinel-1 EW GRDM!
    T have been found empirically. Generally suitable values are around 3e4 - 3.5e4
    Suitable ENL values can be found here:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-detected/ew
    I am generally using ENL around 9-10.
    Known issues:
        -  Large continous objects not detected (typically problems start to occur for objects larger than 8 pixels)
        -  Missed detections in heteorogeneous areas
        -  False detections on edges (eg. between land and water)
        -  Objects cannot be separated (eg. ships/icebergs/islands are all detected the same)
        -  Objects embedded in rough sea ice is hard to detect
    TODO:
        -  Handle NaNs in the area near icebergs (e.g. from landmask)
        - Maybe make a check function to avoid copying all the checks

    Parameters:
    ----------
    img : numpy.ndarray(float32) (K,X,Y)
    mask : numpy.ndarray(bool) (X,Y)
    pfa : float
    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
    """

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    # TODO: print more stuff if error is raied. e.g. print the shape if shape error
    # check datatypes are correct
    if not isinstance(image, np.ndarray) | (image.dtype != np.float32):
        raise TypeError('Input image must be of type np.ndarray(float32)')
    if not isinstance(image, np.ndarray) | (mask.dtype != np.bool):
        raise TypeError('Input mask must be of type np.ndarray(bool)')

    # check if shapes are correct
    if len(image.shape) != 3:
        raise ValueError('Input image must be of shape [K, X, Y]')
    if image.shape[0] != 2:
        raise ValueError('Input image must contain exactly 2 bands')
    if image.shape[1:] != mask.shape:
        raise ValueError('Number of rows/cols must be the same for image and mask')

    # check if the image format
    if smells_like(image) != 'intensity':
        warnings.warn(f'Input image should be in intensity scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    n = 1 * enl     # no looks center pixel
    m = 120 * enl   # no looks edge cell
    p = 2           # no dimensions (bands) in image

    eps = db2in(-100)   # just a small value

    S11_s = np.where(mask, n * image[0, ...], np.nan)  # S11 Ship
    S22_s = np.where(mask, n * image[1, ...], np.nan)  # S22 Ship
    S11_o = m * fast_edge_mean(image[0, ...], mask)  # S11 Ocean
    S22_o = m * fast_edge_mean(image[1, ...], mask)  # S22 Ocean

    S11_o = np.where(S11_o == 0, eps, S11_o)
    S22_o = np.where(S22_o == 0, eps, S22_o)

    detX = S11_s * S22_s
    detY = S11_o * S22_o
    detXY = (S11_s + S11_o) * (S22_s + S22_o)

    lnk = p * ((m + n) * np.log(m + n) - m * np.log(m) - n * np.log(n))
    lnΔ = (n * np.log(detX) + m * np.log(detY)) - ((n + m) * np.log(detXY))
    lnQ = lnk + lnΔ

    P = _calc_prob(p, n, m, lnQ)
    outliers = (1 - P) <= pfa

    # we are only interested in bright outliers
    bright_filter = (S11_o / m < S11_s / n) & (S22_o / m < S22_s / n)
    return (outliers * bright_filter)
