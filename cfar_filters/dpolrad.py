import warnings
import scipy
import numpy as np
from .utils import smells_like
from .fast_functions import test_window, train_window

def _gengamma_minimize(t, gengamma_params, pde):
    return np.abs(scipy.stats.gengamma.cdf(t, *gengamma_params) - pde)

def transform(image, mask=0):
    """
    Dual-Pol Ratio Anomaly Detector Transformation (DPolRAD)
    this function implements equation 2 in the paper
    A. Marino, W. Dierking, and C. Wesche,
    “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,”
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602-5615, 2016, doi: 10.1109/TGRS.2016.2569450.

    The test and training window are 3x3, 57x57 pixels respectively

    Parameters:
    ----------
    image : numpy.ndarray(float32) (2,X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.

    Returns:
    ----------
    transform : numpy.ndarray(float32) (X,Y)
        DPolRad Transform
    """

    HV_target = test_window(image[1, ...], mask)  # test window
    HV_clutter = train_window(image[1, ...], mask)  # train window
    HH_clutter = train_window(image[0, ...], mask)  # train window

    HH_clutter = np.where(HH_clutter == 0, np.nan, HH_clutter)  # dont divide by 0
    transform = (HV_target - HV_clutter) / (HH_clutter)

    return transform

def detector(image, mask=0, pfa=1e-12):
    """
    HV Intensity Dual-Pol Ratio Anomaly Detector (IDPolRAD)

    The detector is based on the transform suggested in the paper:
    A. Marino, W. Dierking, and C. Wesche,
    “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,”
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602-5615, 2016, doi: 10.1109/TGRS.2016.2569450.

    Outliers are detected in the transform using the method outlined in this paper:
    I. H. Soldal, W. Dierking, A. A. Korosov, and A. Marino,
    “Automatic detection of small icebergs in fast ice using satellite Wide-Swath SAR images,”
    Remote Sens., vol. 11, no. 7, pp. 1-24, 2019, doi: 10.3390/rs11070806.

    Specifically, we are applying a global CFAR detector based on the generalized gamma pdf (Table 10 in Soldal)

    Parameters:
    ----------
    image : numpy.ndarray(float32) (2,X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    pfa : float
        Probability of false alarm. Should be somewhere between 0-1

    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
        Binary outlier classification
    """

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    # check datatypes are correct
    if (not isinstance(image, np.ndarray)) | (image.dtype != np.float32):
        raise TypeError(f'Input image must be of type np.ndarray(float32) but is of type {type(image)}, {image.dtype}')
    if (not isinstance(mask, np.ndarray)) | (mask.dtype != np.bool):
        raise TypeError(f'Input mask must be of type np.ndarray(bool) but is of type {type(mask)}, {mask.dtype}')

    # check if shapes are correct
    if (len(image.shape) != 3) or (image.shape[0] != 2):
        raise ValueError(f'Input image must be of shape [2, X, Y] but is of shape {image.shape}')
    if image.shape[1:] != mask.shape:
        raise ValueError((f'Shape of mask must match shape of image. \
                          Mask shape: {mask.shape}. Image shape {image.shape[1:]}'))

    # check if the image format
    if smells_like(image) != 'intensity':
        warnings.warn(f'Input image should be in intensity scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    # calculate the dpolrad transform
    dpolrad = transform(image, mask=mask)
    # multiply with HV to get contrast enhancement
    idpolrad = image[1, ...] * dpolrad
    # if dpolrad is negative, Idpolrad is set to zero (A. Marino Sec. IV C)
    idpolrad = np.where(dpolrad < 0, 0, idpolrad)
    # if nan set to zero also
    idpolrad = np.where(np.isnan(idpolrad), 0, idpolrad)

    # estimate the parameters for the generalized gamma distribution
    # exclude values smaller than 50 x mean and larger than 0
    # using all samples is very slow - so we estimate the parameters based on a random subset
    N = 10000
    np.random.seed(0)  # seed random state so same pixels are used each time
    samples = np.random.choice(idpolrad[np.where((idpolrad < 50 * np.mean(idpolrad)) & (idpolrad > 0))], N)
    gengamma_params = scipy.stats.gengamma.fit(samples)

    # calculate the threshold corresponding to the desired PFA
    pde = 1 - pfa
    init = 10 * np.mean(idpolrad)  # initial guess of the algorithm
    T = scipy.optimize.fmin(_gengamma_minimize, init, disp=False, args=(gengamma_params, pde), ftol=1e-21)[0]

    outliers = (idpolrad > T)

    return outliers
