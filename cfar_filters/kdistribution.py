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
def _k_minimize(t, μ, v, enl, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(enl)))[0] - pde)

def _k_params(image, enl):
    vmin = 1
    vmax = 50

    μ = np.nanmean(image)
    K = np.nanmean(image**2) / μ**2
    v = (enl + 1) / (enl * K - enl - 1)

    v = min(max(vmin, v), vmax)

    return μ, v

# K-distribution CFAR on image blocks
def _kd_cfar(image, μ, v, enl, pde):
    init = 5 * μ  # initial guess of the algorithm - set empirically
    T = fmin(_k_minimize, init, disp=False, args=(μ, v, enl, pde))[0]
    outliers = image > T
    return outliers

def detector(image, bs=500, pfa=1e-6, enl=10.7):

    outliers = np.zeros_like(image).astype(np.bool)
    pde = 1 - pfa  # probability of detection

    n_rows, n_cols = np.asarray(image.shape[0:2]) // bs + 1

    for x in range(0, n_rows):
        for y in range(0, n_cols):
            sub_block_image = image[x * bs:x * bs + bs, y * bs:y * bs + bs]
            μ, v = _k_params(sub_block_image, enl)
            outliers[x * bs:x * bs + bs, y * bs:y * bs + bs] = _kd_cfar(sub_block_image, μ, v, enl, pde)

    return outliers
