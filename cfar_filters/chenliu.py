import numpy as np
from .fast_functions import fast_edge_mean

# TODO: Maybe we should remane this file. chenliu is too generic, maybe rename to normsum
def transform(image, mask=0):
    """
    Normalized Intensity Sum transformation.
    Based on the normalized intensity sum decision variable suggested by Chen Liu in:
    Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

    Specifically, this function implements e.q. 10 in the paper

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

    HH_edge = fast_edge_mean(image[0, ...], mask)
    HV_edge = fast_edge_mean(image[1, ...], mask)

    HH_edge = np.where(HH_edge == 0, np.nan, HH_edge)
    HV_edge = np.where(HV_edge == 0, np.nan, HV_edge)

    NIS = (image[0, ...] / HH_edge) + (image[1, ...] / HV_edge)

    return NIS
