
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


# # Corresponding to outer_window_size==15
# @nb.stencil(neighborhood=((-7, 7), (-7, 7)))
# def _edge_kernel_mean(x, m):
#     if m[0, 0]:
#         cumul = 0
#         for i in range(-7, 8):
#             for ii in range(-7, 8):
#                 # Corresponding to inner_window_size==9
#                 if (i < -4 or i > 4) or (ii < -4 or ii > 4):
#                     cumul += x[i, ii]
#         return nb.float32(cumul / 144)
#     else:
#         return nb.float32(np.nan)

@nb.stencil(neighborhood=((-20, 20), (-20, 20)))
def _edge_kernel_mean(x, m, ri, ro):

    if m[0, 0]:  # if pixel is not masked
        cumul = 0
        no_samples = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                # if pixel within the radius limits and not nan
                if (ri < np.sqrt(i**2 + ii**2) < ro) & ~np.isnan(x[i, ii]):
                    cumul += x[i, ii]
                    no_samples += 1
        if no_samples == 0:
            return nb.float32(0)
        else:
            return nb.float32(cumul / no_samples)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==15
@nb.stencil(neighborhood=((-20, 20), (-20, 20)))
def _edge_kernel_std(x, m, ri, ro):

    if m[0, 0]:  # if pixel is not masked
        cumul = 0
        no_samples = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                # if pixel within the radius limits and not nan
                if (ri < np.sqrt(i**2 + ii**2) < ro) & ~np.isnan(x[i, ii]):
                    cumul += x[i, ii]
                    no_samples += 1
        mean = nb.float32(cumul / no_samples)

        cumul = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                if (ri < np.sqrt(i**2 + ii**2) < ro) & ~np.isnan(x[i, ii]):
                    cumul += (x[i, ii] - mean)**2
        return nb.float32(np.sqrt(cumul / no_samples))
    else:
        return nb.float32(np.nan)

# # Corresponding to test window of size 3
# @nb.stencil(neighborhood=((-1, 1), (-1, 1)))
# def _test_window_mean(x, m):
#     if m[0, 0]:
#         cumul = 0
#         for i in range(-1, 2):
#             for ii in range(-1, 2):
#                 cumul += x[i, ii]
#         return nb.float32(cumul / 9)
#     else:
#         return nb.float32(np.nan)

# # Corresponding to a training window of size 57
# @nb.stencil(neighborhood=((-28, 28), (-28, 28)))
# def _train_window_mean(x, m):
#     if m[0, 0]:
#         cumul = 0
#         for i in range(-28, 29):
#             for ii in range(-28, 29):
#                 cumul += x[i, ii]
#         return nb.float32(cumul / 3249)
#     else:
#         return nb.float32(np.nan)

# @nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
# def test_window(x, m):
#     return _test_window_mean(x, m)

# @nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
# def train_window(x, m):
#     return _train_window_mean(x, m)

# @nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
# def fast_edge_mean(x, m):
#     return _edge_kernel_mean(x, m)


@nb.jit('float32[:,:](float32[:,:], boolean[:,:], u2, u2)', parallel=True, nopython=True)
def fast_edge_mean(x, m, ri, ro):
    return _edge_kernel_mean(x, m, ri, ro)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:], u2, u2)', parallel=True, nopython=True)
def fast_edge_std(x, m, ri, ro):
    return _edge_kernel_std(x, m, ri, ro)
