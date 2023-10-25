import warnings
import numpy as np
from .utils import smells_like, mask_edges
from .fast_functions import fast_edge_mean

def transform(image, mask=0, wi=9, wo=15):
    """
    Normalized Intensity Sum transformation (NIS).
    Based on the normalized intensity sum decision variable suggested by Chen Liu in:
    Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

    Specifically, this function implements e.q. 10 in the paper.

    This method is similar to the polarimetric whitening filter (PWF).

    Parameters:
    ----------
    image : numpy.ndarray(float32) (2,X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    wi : integer
        Inner Window Diameter - also called guard area
    wo : integer
        Outer Window Diameter - also called clutter estimation area

    Returns:
    ----------
    transform : numpy.ndarray(float32) (X,Y)
        NIS Transform
    """

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

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

    HH_edge = fast_edge_mean(image[0, ...], mask, wi, wo)
    HV_edge = fast_edge_mean(image[1, ...], mask, wi, wo)

    HH_edge = np.where(HH_edge == 0, np.nan, HH_edge)
    HV_edge = np.where(HV_edge == 0, np.nan, HV_edge)

    NIS = (image[0, ...] / HH_edge) + (image[1, ...] / HV_edge)

    transform = mask_edges(NIS, 20, np.nan)

    return transform
