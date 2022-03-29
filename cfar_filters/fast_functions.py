
"""
Fast convolutions using NUMBA

Kernel sizes made for 40 meter SAR images
Inner window of 9 (~360m)
Outer window of 15 (~520m)
This corresponds to totally 15**2 - 9**2 = 144 pixels in outer window

Icebergs at Labrador sea is on average 130 meters long. The maximum icebergs are 379 meters long, 
with 75% of the icebergs being below 160 meters. If we design the filter for 160 meter icebergs, 
the inner window whould be 320 or ~ 8 pixels. We need to round up to 9 or ~360m. The outer window 
of 15 pixels ensure that we have more than 140 pixels for estimating the clutter. 
Larger outer window will mean slower execution.



"""

import numpy as np
import numba as nb

# Corresponding to inner_window_size==5
@nb.stencil(neighborhood=((-2, 2), (-2, 2)))
def _inner_kernel_mean(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-2, 3):
            for ii in range(-2, 3):
                cumul += x[i, ii]
        return nb.float32(cumul / 25)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==15
@nb.stencil(neighborhood=((-7, 7), (-7, 7)))
def _outer_kernel_mean(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-7, 8):
            for ii in range(-7, 8):
                cumul += x[i, ii]
        return nb.float32(cumul / 225)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==15
@nb.stencil(neighborhood=((-7, 7), (-7, 7)))
def _edge_kernel_mean(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-7, 8):
            for ii in range(-7, 8):
                # Corresponding to inner_window_size==9
                if (i < -4 or i > 4) or (ii < -4 or ii > 4):
                    cumul += x[i, ii]
        return nb.float32(cumul / 144)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==15
@nb.stencil(neighborhood=((-7, 7), (-7, 7)))
def _edge_kernel_std(x, m):
    if m[0, 0]:
        cumul = 0
        for i in range(-7, 8):
            for ii in range(-7, 8):
                # Corresponding to inner_window_size==9
                if (i < -4 or i > 4) or (ii < -4 or ii > 4):
                    cumul += x[i, ii]
        mean = nb.float32(cumul / 144)

        cumul = 0
        for i in range(-7, 8):
            for ii in range(-7, 8):
                if (i < -4 or i > 4) or (ii < -4 or ii > 4):
                    cumul += (x[i, ii] - mean)**2
        return nb.float32(np.sqrt(cumul / 144))
    else:
        return nb.float32(np.nan)


@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_mean(x, m):
    return _edge_kernel_mean(x, m)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_std(x, m):
    return _edge_kernel_std(x, m)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_inner_mean(x, m):
    return _inner_kernel_mean(x, m)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_outer_mean(x, m):
    return _outer_kernel_mean(x, m)
