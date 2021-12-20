import numpy as np
from scipy.special import gamma, kn
import scipy.integrate as integrate
from scipy.optimize import fmin

# implementation of the K-distribution pdf
# from Quegan eq. 8.21
# This expression also use by Brekke. And it is identical to the expression used by Liu
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

# wrapper for the integration function
# the function becomes numerically instable when enl and v are floating
# I dont know what causes this, but choose to round as that seems like the best approach
# def _k_integration(I, μ, v, enl):
#     return k_pdf(I, μ, np.round(v), np.round(enl))

# Numerical integration of the k-distribution using scipy.integrate
def _k_minimize(t, μ, v, L, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(L)))[0] - pde)

def _k_params(image, L):
    vmin = 1
    vmax = 50

    μ = np.nanmean(image)
    K = np.nanmean(image**2) / μ**2
    v = (L + 1) / (L * K - L - 1)
    v = min(max(vmin, v), vmax)

    return μ, v

# K-distribution CFAR on image blocks
def _kd_cfar(image, μ, v, L, pde):
    init = 5 * μ  # initial guess of the algorithm
    T = fmin(_k_minimize, init, disp=False, args=(μ, v, L, pde))[0]
    outliers = image > T
    return outliers


def detector(image, N=500, pfa=1e-12, L=np.nan):

    # if we dont have L, use ENL estimation
    est_enl_flag = False
    if np.isnan(L):
        est_enl_flag = True

    outliers = np.zeros_like(image).astype(np.bool)
    pde = 1 - (pfa)  # probability of detection

    n_rows, n_cols = np.asarray(image.shape[0:2]) // N + (np.mod(np.asarray(image.shape[0:2]), N) > 0) * 1

    for x in range(0, n_rows):
        for y in range(0, n_cols):

            sub_block_image = image[x * N:x * N + N, y * N:y * N + N]
            sub_block_image = sub_block_image - np.nanmin(sub_block_image)  # make sure posistive

            if est_enl_flag:
                L = np.nanmean(sub_block_image)**2 / np.nanvar(sub_block_image)

            μ, v = _k_params(sub_block_image, L)

            outliers[x * N:x * N + N, y * N:y * N + N] = _kd_cfar(sub_block_image, μ, v, L, pde)

    return outliers
