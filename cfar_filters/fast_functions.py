
import numpy as np
import numba as nb

# Corresponding to inner_window_size==5
@nb.stencil( neighborhood = ((-2, 2),(-2, 2)))
def _center_kernel_mean(x):
    cumul = 0
    for i in range(-2,3):
        for ii in range(-2,3):
            cumul += x[i,ii]
    return nb.float32(cumul / 25)


# Corresponding to outer_window_size==21
@nb.stencil( neighborhood = ((-10, 10),(-10, 10)))
def _edge_kernel_mean(x):
    cumul = 0
    for i in range(-10,11):
        for ii in range(-10,11):
            # Corresponding to middle_window_size==15
            if (i<=-7 or i>=7) or (ii<=-7 or ii>=7):
                cumul += x[i,ii]
    return nb.float32(cumul / 216)


# Corresponding to outer_window_size==21
@nb.stencil( neighborhood = ((-10, 11),(-10, 11)))
def _edge_kernel_std(x):
    cumul = 0
    for i in range(-10,11):
        for ii in range(-10,11):
            # Corresponding to middle_window_size==15
            if (i<=-7 or i>=7) or (ii<=-7 or ii>=7):
                cumul += x[i,ii]
    mean = nb.float32(cumul / 216)

    cumul = 0
    for i in range(-10,11):
        for ii in range(-10,11):
            if (i<=-7 or i>=7) or (ii<=-7 or ii>=7):
                cumul += (x[i,ii]-mean)**2
    return nb.float32(np.sqrt(cumul/215))


@nb.jit('float32[:,:](float32[:,:])', parallel=True, nopython=True)
def fast_center_mean(x):
    return _center_kernel_mean(x)


@nb.jit('float32[:,:](float32[:,:])', parallel=True, nopython=True)
def fast_edge_mean(x):
    return _edge_kernel_mean(x)


@nb.jit('float32[:,:](float32[:,:])', parallel=True, nopython=True)
def fast_edge_std(x):
    return _edge_kernel_std(x)


