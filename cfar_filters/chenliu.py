import numpy as np
from .fast_functions import fast_edge_mean

def transform(image, mask=0, pfa=1e-6):

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    HH_edge = fast_edge_mean(image[0, ...], mask)
    HV_edge = fast_edge_mean(image[1, ...], mask)

    HH_edge = np.where(HH_edge == 0, np.nan, HH_edge)
    HV_edge = np.where(HV_edge == 0, np.nan, HV_edge)

    detection_image = (image[0, ...] / HH_edge) + (image[1, ...] / HV_edge)

    return detection_image
