import numpy as np
from rs_utils.sar_functions import db2in
from .fast_functions import fast_edge_mean, fast_center_mean


def transform(image, mask=0, pfa=1e-6):

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    eps = db2in(-100)

    HV_test = fast_center_mean(image[1, ...],)
    HV_train = fast_edge_mean(image[1, ...])
    HH_train = fast_edge_mean(image[0, ...])

    Δ = (HV_test - HV_train) / (HH_train + eps) > HV_test

    detection_image = (Δ * image[1, ...]**2 + eps)

    return detection_image
