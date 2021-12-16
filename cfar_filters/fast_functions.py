
"""
Fast convolutions using NUMBA

Kernel sizes made for 40 meter SAR images
Inner window of 7 (~280m)
Outer window of 13 (~520m)
This corresponds to totally 13**2 - 7**2 = 120 pixels in outer window

"""

import numpy as np
import numba as nb

# Corresponding to inner_window_size==3
@nb.stencil(neighborhood=((-1, 1), (-1, 1)))
def _center_kernel_mean(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-1, 2):
            for ii in range(-1, 2):
                cumul += x[i, ii]
        return nb.float32(cumul / 9)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==13
@nb.stencil(neighborhood=((-6, 6), (-6, 6)))
def _edge_kernel_mean(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-6, 7):
            for ii in range(-6, 7):
                # Corresponding to inner_window_size==7
                if (i <= -3 or i >= 3) or (ii <= -3 or ii >= 3):
                    cumul += x[i, ii]
        return nb.float32(cumul / 120)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==13
@nb.stencil(neighborhood=((-6, 6), (-6, 6)))
def _edge_kernel_std(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-6, 7):
            for ii in range(-6, 7):
                # Corresponding to inner_window_size==7
                if (i <= -3 or i >= 3) or (ii <= -3 or ii >= 3):
                    cumul += x[i, ii]
        mean = nb.float32(cumul / 120)

        cumul = 0
        for i in range(-6, 7):
            for ii in range(-6, 7):
                if (i <= -3 or i >= 3) or (ii <= -3 or ii >= 3):
                    cumul += (x[i, ii] - mean)**2
        return nb.float32(np.sqrt(cumul / 120))
    else:
        return nb.float32(np.nan)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_center_mean(x, m):
    return _center_kernel_mean(x, m)


@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_mean(x, m):
    return _edge_kernel_mean(x, m)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_std(x, m):
    return _edge_kernel_std(x, m)

# 4-connected only, change if we use 8-connected objects
@nb.stencil(neighborhood=((-1, 1), (-1, 1)))
def _count_connected(x):
    cumul = 0
    for i in range(-1, 2):
        if i != 0:
            cumul += x[i, 0] * 1
            cumul += x[0, i] * 1
    return cumul

@nb.jit('int8(boolean[:,:])', parallel=True, nopython=True)
def total_connections(x):
    return nb.int8(np.sum(_count_connected(x) * x) / 2)

# calculate the outer perimeter of a binary object
def outer_perimeter(regionmask):
    N = np.sum(regionmask)
    C = total_connections(np.pad(regionmask, pad_width=1))
    return np.int8(4 * N - 2 * C)
