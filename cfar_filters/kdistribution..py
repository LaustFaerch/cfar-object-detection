import numpy as np

from scipy.special import gamma, kn
import scipy.integrate as integrate
from scipy.optimize import fmin

# implementation of the K-distribution pdf
# from Quegan eq. 8.21
# This expression also use by Brekke. And it is identical to the expression used by Liu
def k_pdf(I, L, v, μ):

    v = np.float64(v)
    I = np.float64(I)
    n = np.round(v-L) # order of bessel function must be integer
    
    a = 2/(gamma(L)*gamma(v))
    b = (L*v/μ)**((L+v)/2)
    c = I**((L+v-2)/2)
    d = kn(n, 2*np.sqrt(L*v*I/μ))
    
    pdf = a*b*c*d
    
    pdf = np.where(np.isnan(pdf), 0, pdf)
    
    return pdf

# wrapper for the integration function
# the function becomes numerically instable when enl and v are floating
# I dont know what causes this, but choose to round as that seems like the best approach
def _k_integration(I):
    return k_pdf(I, np.round(enl), np.round(v), μ)

# Numerical integration of the k-distribution using scipy.integrate
def _k_minimize(t):
    return np.abs(integrate.quad(_k_integration, 0, t)[0]-pd)

def _k_params(image):
    vmin = 1
    vmax = 50

    μ = np.nanmean(image)
    K = np.nanmean(image**2) / μ**2
    v = (enl+1)/(enl*K-enl-1)
    
    v = min(max(vmin, v), vmax)
    
    return μ, K, v

# K-distribution CFAR on image blocks
def _kd_cfar(image):
    init = 5*μ # initial guess of the algorithm
    T = fmin(_k_minimize, init, disp=False)[0]
    outliers = image>T
    return outliers, T

def detector(image, blocksize=500, pfa=1e-6, enl=10.7):

    outliers = np.zeros_like(image).astype(np.bool)
    threshold = np.zeros_like(image)
    pd = 1-(pfa)

    n_rows, n_cols = np.asarray(image.shape[0:2]) // blocksize + 1

    for x in range(0, n_rows):
        for y in range(0, n_cols):
            sub_block_image = image[x*blocksize:x*blocksize+blocksize, y*blocksize:y*blocksize+blocksize]
            μ, K, v = _k_params(sub_block_image) # global params
            outliers[x*blocksize:x*blocksize+blocksize, y*blocksize:y*blocksize+blocksize] = _kd_cfar(sub_block_image)

    return outliers