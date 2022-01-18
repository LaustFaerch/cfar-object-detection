import warnings
import numpy as np
from .utils import smells_like, mask_edges
from .fast_functions import fast_edge_mean, fast_center_mean


def transform(image, mask=0):
    """
    Depolarization Ratio Anomaly Transformation.
    Based on the following paper:
    A. Marino, W. Dierking, and C. Wesche,
    “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,”
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602-5615, 2016, doi: 10.1109/TGRS.2016.2569450.

    Specifically, this function implements equation 2 in the paper.

    Parameters:
    ----------
    image : numpy.ndarray(float32) (2, X,Y)
        SAR image in linear intensity formal
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
    if not isinstance(image, np.ndarray) | (image.dtype != np.float32):
        raise TypeError(f'Input image must be of type np.ndarray(float32) but is of type {type(image)}, {image.dtype}')
    if not isinstance(mask, np.ndarray) | (mask.dtype != np.bool):
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

    HV_target = fast_center_mean(image[1, ...], mask)
    HV_clutter = fast_edge_mean(image[1, ...], mask)
    HH_clutter = fast_edge_mean(image[0, ...], mask)

    HH_clutter = np.where(HH_clutter == 0, np.nan, HH_clutter)
    Δ = (HV_target - HV_clutter) / (HH_clutter)

    Δ = mask_edges(Δ, 6, np.nan)

    return Δ
