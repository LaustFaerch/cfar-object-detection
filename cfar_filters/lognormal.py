import warnings
import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
from rs_utils.sar_functions import smells_like
from .fast_functions import fast_edge_mean, fast_edge_std


def _gaussian_pfa(t):
    return 0.5 - 0.5 * erf(t / np.sqrt(2))

def _gaussian_pfa_minimization(x, pfa):
    return np.abs(_gaussian_pfa(x) - pfa)

def _find_gaussian_multiplier(pfa):
    res = minimize(_gaussian_pfa_minimization, 5, args=(pfa), method='Nelder-Mead', tol=1e-6)
    return res.x[0]

def detector(image, mask=0, pfa=1e-6):

    # if no mask is given, assume all pixels are valid
    if np.all(mask == 0):
        mask = np.ones_like(image[0, ...]) > 0

    image = image.squeeze()
    # Define filter parameters
    std_dev_multiplier = _find_gaussian_multiplier(pfa)

    edge_mean = fast_edge_mean(image)
    egde_std = fast_edge_std(image)

    outliers = image > (edge_mean + std_dev_multiplier * egde_std)

    return outliers
