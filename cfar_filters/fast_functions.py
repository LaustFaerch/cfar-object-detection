
import numpy as np
import numba as nb

# Corresponding to inner_window_size==5
@nb.stencil( neighborhood = ((-2, 2),(-2, 2)))
def _center_kernel_mean(x,m):
    if m[0, 0]:
        cumul = 0
        for i in range(-2,3):
            for ii in range(-2,3):
                cumul += x[i,ii]
        return nb.float32(cumul / 25)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==21
@nb.stencil( neighborhood = ((-10, 10),(-10, 10)))
def _edge_kernel_mean(x,m):
    if m[0, 0]:
        cumul = 0
        for i in range(-10,11):
            for ii in range(-10,11):
                # Corresponding to middle_window_size==15
                if (i<=-7 or i>=7) or (ii<=-7 or ii>=7):
                    cumul += x[i,ii]
        return nb.float32(cumul / 216)
    else:
        return nb.float32(np.nan)

# Corresponding to outer_window_size==21
@nb.stencil( neighborhood = ((-10, 11),(-10, 11)))
def _edge_kernel_std(x,m):
    if m[0, 0]:
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
    else:
        return nb.float32(np.nan)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_center_mean(x,m):
    return _center_kernel_mean(x,m)


@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_mean(x,m):
    return _edge_kernel_mean(x,m)


@nb.jit('float32[:,:](float32[:,:], boolean[:,:])', parallel=True, nopython=True)
def fast_edge_std(x,m):
    return _edge_kernel_std(x,m)

# 4-connected only, change if we use 8-connected objects
@nb.stencil( neighborhood = ((-1, 1),(-1, 1)))
def _count_connected(x):
    cumul = 0
    for i in range(-1,2):
        if i!=0:
            cumul += x[i,0]*1
            cumul += x[0,i]*1
    return cumul

@nb.jit('int8(boolean[:,:])', parallel=True, nopython=True)
def total_connections(x):
    return nb.int8(np.sum(_count_connected(x)*x)/2)


# calculate the outer perimeter of a binary object
def outer_perimeter(regionmask):
    N = np.sum(regionmask)
    C = total_connections(np.pad(regionmask, pad_width=1))
    return np.int8(4*N-2*C)