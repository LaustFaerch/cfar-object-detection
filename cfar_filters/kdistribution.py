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

# Numerical integration of the k-distribution using scipy.integrate
def _k_minimize(t, μ, v, L, pde):
    return np.abs(integrate.quad(k_pdf, 0, t, args=(μ, np.round(v), np.round(L)))[0] - pde)

def _mom_estimation_full(image):

    median = np.nanmedian(image)
    image = np.where(image > 3 * median, np.nan, image)

    μ = np.nanmean(image)

    I2 = np.nanmean((image)**2)  # 2nd raw sample moment
    I3 = np.nanmean((image)**3)  # 3rd raw sample moment

    W2 = I2 / μ**2
    W3 = (I3 / μ**3) / (W2)

    b = (W3 - 4 * W2 - 1)
    D = b**2 - 16 * W2

    # no solution
    if D <= 0:
        return μ, np.nan, np.nan

    Vplus = 1 / 4 * (-b + np.sqrt(D))
    Vminus = 1 / 4 * (-b - np.sqrt(D))
    V = max(Vplus, Vminus)
    U = W2 / V

    L = 1 / (U - 1)
    v = 1 / (V - 1)

    return μ, v, L

def _mom_estimation_simple(image):

    μ = np.nanmean(image)
    L = μ**2 / np.nanvar(image)
    K = np.nanmean(image**2) / μ**2
    v = (L + 1) / (L * K - L - 1)

    return μ, v, L

# K-distribution CFAR on image blocks
def _kd_cfar(image, μ, v, L, pde):
    init = 5 * μ  # initial guess of the algorithm
    T = fmin(_k_minimize, init, disp=False, args=(μ, v, L, pde))[0]
    outliers = image > T
    return outliers


def detector(image, N=250, pfa=1e-12, offset=False):

    vmin, vmax = 1, 50
    Lmin, Lmax = 1, 22  # 2xENL

    outliers = np.zeros_like(image).astype(np.bool)
    pde = 1 - (pfa)  # probability of detection

    n_rows, n_cols = np.asarray(image.shape[0:2]) // N + (np.mod(np.asarray(image.shape[0:2]), N) > 0) * 1

    for x in range(0, n_rows):
        for y in range(0, n_cols):

            sub_block_image = image[x * N:x * N + N, y * N:y * N + N]

            # if block is masked then skip the block
            if np.all(np.isnan(sub_block_image)) or np.all(sub_block_image == 0):
                outliers[x * N:x * N + N, y * N:y * N + N] = np.zeros_like(sub_block_image) > 0
            else:

                if offset:
                    # offset to ensure pdf starts near 0 (which is important for the mom estimation)
                    # use offset for dpolrad transform, NOT for the chenliu transform
                    sub_block_image = sub_block_image - np.nanmin(sub_block_image)

                # ESTIMATE PARAMETERS FOR THE K-DISTRIBUTION
                μ, v, L = _mom_estimation_full(sub_block_image)

                # if the mom estimation fails, use a simpler estimation
                if np.any(np.isnan(np.array([v, L]))):
                    μ, v, L = _mom_estimation_simple(sub_block_image)

                # If v is negative it is likely that the equations broke down
                # due to the denominator in the simple mom estimation
                # If v is very large, the distribution is likely near gamma
                # In both cases use vmax.
                if v <= vmin or v >= vmax:
                    v = vmax
                # L cannot be negative
                if L <= 0:
                    L = Lmax
                # L cannot be smaller than 1 or larger than p*ENL
                L = min(max(Lmin, L), Lmax)

                outliers[x * N:x * N + N, y * N:y * N + N] = _kd_cfar(sub_block_image, μ, v, L, pde)

    return outliers
