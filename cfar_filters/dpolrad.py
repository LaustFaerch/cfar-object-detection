import warnings
import numpy as np
from .utils import smells_like, mask_edges
from .fast_functions import fast_inner_mean, fast_outer_mean


def transform(image, mask=0):
    """
    Dual-Pol Ratio Anomaly Detector Transformation (DPolRAD)
    Based on the following paper:
    A. Marino, W. Dierking, and C. Wesche,
    “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,”
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602-5615, 2016, doi: 10.1109/TGRS.2016.2569450.

    Specifically, this function implements equation 2 in the paper.

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

    HV_target = fast_inner_mean(image[1, ...], mask)  # test window
    HV_clutter = fast_outer_mean(image[1, ...], mask)  # train window
    HH_clutter = fast_outer_mean(image[0, ...], mask)  # train window

    HH_clutter = np.where(HH_clutter == 0, np.nan, HH_clutter)
    transform = HV_target * (HV_target - HV_clutter) / (HH_clutter)

    transform = mask_edges(transform, 6, np.nan)

    return transform
