
"""
Fast convolutions using NUMBA

Window sizes can be varied using the wi/wo parameters, but maximum allowable window sie is 41x41 pixels.
This might not be suitable for e.g., very high resolution SAR images.
It might be nescessary to change the window sizes for some applications (e.g., very high res images).
However, increasing the window sizes will increase computational cost with O^2.

"""

import numpy as np
import numba as nb

@nb.stencil(neighborhood=((-20, 20), (-20, 20)))
def _edge_kernel_mean(x, m, ri, ro):

    if m[0, 0]:  # if pixel is not masked
        cumul = 0
        no_samples = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                # if pixel within the radius limits and not nan
                if (ri <= np.sqrt(i**2 + ii**2) <= ro) & ~np.isnan(x[i, ii]):
                    cumul += x[i, ii]
                    no_samples += 1
        if no_samples == 0:
            return nb.float32(0)
        else:
            return nb.float32(cumul / no_samples)
    else:
        return nb.float32(np.nan)

@nb.stencil(neighborhood=((-20, 20), (-20, 20)))
def _edge_kernel_std(x, m, ri, ro):

    if m[0, 0]:  # if pixel is not masked
        cumul = 0
        no_samples = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                # if pixel within the radius limits and not nan
                if (ri <= np.sqrt(i**2 + ii**2) <= ro) & ~np.isnan(x[i, ii]):
                    cumul += x[i, ii]
                    no_samples += 1
        mean = nb.float32(cumul / no_samples)

        cumul = 0
        for i in range(-20, 21):
            for ii in range(-20, 21):
                if (ri <= np.sqrt(i**2 + ii**2) <= ro) & ~np.isnan(x[i, ii]):
                    cumul += (x[i, ii] - mean)**2
        return nb.float32(np.sqrt(cumul / no_samples))
    else:
        return nb.float32(np.nan)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:], u2, u2)', parallel=True, nopython=True)
def fast_edge_mean(x, m, wi, wo):
    """
        Parameters:
        ----------
        x : numpy.ndarray(float32) (X,Y)
            SAR image
        m : numpy.ndarray(bool) (X,Y)
            Mask for the image.
        wi : integer
            Inner Window Diameter - also called guard area
        wo : integer
            Outer Window Diameter - also called clutter estimation area

        Returns:
        ----------
        outliers : numpy.ndarray(float32) (X,Y)
            Running average using the window shape defined by wi and wo.
    """
    ri, ro = wi // 2, wo // 2
    return _edge_kernel_mean(x, m, ri, ro)

@nb.jit('float32[:,:](float32[:,:], boolean[:,:], u2, u2)', parallel=True, nopython=True)
def fast_edge_std(x, m, wi, wo):
    """
        Parameters:
        ----------
        x : numpy.ndarray(float32) (X,Y)
            SAR image
        m : numpy.ndarray(bool) (X,Y)
            Mask for the image.
        wi : integer
            Inner Window Diameter - also called guard area
        wo : integer
            Outer Window Diameter - also called clutter estimation area

        Returns:
        ----------
        outliers : numpy.ndarray(float32) (X,Y)
            Running std. using the window shape defined by wi and wo.
    """
    ri, ro = wi // 2, wo // 2
    return _edge_kernel_std(x, m, ri, ro)
