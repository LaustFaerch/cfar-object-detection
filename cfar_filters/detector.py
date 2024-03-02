"""
CFAR Detection on dual-band SAR data.

If using this code, please use the following citation:
Færch, L., Dierking, W., Hughes, N., and Doulgeris, A. P.:
A Comparison of CFAR Object Detection Algorithms for Iceberg Identification in
L- and C-band SAR Imagery of the Labrador Sea,
The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2023-17, in review, 2023.


Copyright (C) 2023  Laust Færch

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import warnings
import numpy as np
from .utils import smells_like
from . import kdistribution, lognormal, wishart, normsum, dpolrad, gamma, utils

def run(image, mask, detector='gamma', method='AND', pfa=1e-9, enl=10.7, minsize=2, sensitivity=40, wi=9, wo=15):
    """
    Run CFAR detection on dual-band SAR data

    Parameters:
    ----------
    image : numpy.ndarray(float32) (2,X,Y)
        SAR image in linear intensity format
    mask : numpy.ndarray(bool) (X,Y)
        Mask for the image.
    detector : string
        CFAR detector used. Choose between: \'gamma\', \'lognorm\', \'k\', \'wishart\', \'nis\', \'idpolrad\'.
    method: string
        Method for boolean combination of detectors. Choose between \'AND\' or \'OR\'.
    pfa : float
        Probability of false alarm. Should be in interval (0,1)
    enl : float
        Equivalent Number of Looks of the SAR image. Typically between 5-20 depending on the sensor
    minsize : integer
        Objects this size or smaller are assumed to be noise and are removed.
    sensitiviy : integer
        Number of v-estimations for the K detector LUT. Larger number means slower but more precise calculations.
    wi : integer
        Inner Window Diameter - also called guard area
    wo : integer
        Outer Window Diameter - also called clutter estimation area

    Returns:
    ----------
    outliers : numpy.ndarray(bool) (X,Y)
        Binary outlier classification
    """

    # check if the image format
    if smells_like(image) != 'decibel':
        warnings.warn(f'Input image should be in decibel scale. Image smells like {smells_like(image)}',
                      category=UserWarning)

    # check method is valid
    if method not in ['AND', 'OR']:
        raise ValueError((f'Method not recognized. Method should be either AND or OR, you gave {method}'))
    # check pfa within reasonable limits
    if (pfa <= 0) | (pfa >= 1):
        raise ValueError((f'PFA must be in interval (0, 1), you gave: {pfa}'))

    # check that window sizes are valid
    if wi > wo:
        raise ValueError((f'Outer window must be larger than inner window \
                            wi: {wi}, wo {wo}'))
    if wo > 40:
        warnings.warn((f'Maximum supported window size is 40. You gave wo = {wo} \
                            If you want larger windows, edit the neighbourhood and ranges in fast_functions.\
                            Be aware; complexity increases with the square of the neighborhood size!'),
                      category=UserWarning)

    # check datatypes are correct
    if (not isinstance(image, np.ndarray)) | (image.dtype != np.float32):
        raise TypeError(f'Input image must be of type np.ndarray(float32) but is of type {type(image)}, {image.dtype}')
    if (not isinstance(mask, np.ndarray)) | (mask.dtype != bool):
        raise TypeError(f'Input mask must be of type np.ndarray(bool) but is of type {type(mask)}, {mask.dtype}')

    # check dimensions of image
    if (len(image.shape) == 2):
        raise TypeError('Input image appear to be single-band. For single band images, please call the individual \
                        detector functions directly. e.g.: outliers = gamma.detector(image, mask, pfa, enl, wi, wo)')

    if (len(image.shape) != 3) or (image.shape[0] != 2):
        raise ValueError(f'Input image must be dual-band of shape [2, X, Y] but is of shape {image.shape}')

    if detector == 'gamma':
        image = utils.db2in(image)

        if method == 'AND':
            hh_outliers = gamma.detector(image[0, ...], mask=mask, pfa=np.sqrt(pfa), enl=enl, wi=wi, wo=wo)
            hv_outliers = gamma.detector(image[1, ...], mask=mask, pfa=np.sqrt(pfa), enl=enl, wi=wi, wo=wo)
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = gamma.detector(image[0, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), enl=enl, wi=wi, wo=wo)
            hv_outliers = gamma.detector(image[1, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), enl=enl, wi=wi, wo=wo)
            outliers = hh_outliers | hv_outliers

    elif detector == 'lognorm':

        if method == 'AND':
            hh_outliers = lognormal.detector(image[0, ...], mask=mask, pfa=np.sqrt(pfa), wi=wi, wo=wo)
            hv_outliers = lognormal.detector(image[1, ...], mask=mask, pfa=np.sqrt(pfa), wi=wi, wo=wo)
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = lognormal.detector(image[0, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), wi=wi, wo=wo)
            hv_outliers = lognormal.detector(image[1, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), wi=wi, wo=wo)
            outliers = hh_outliers | hv_outliers

    elif detector == 'k':
        image = utils.db2in(image)

        if method == 'AND':
            hh_outliers = kdistribution.detector(image[0, ...], mask=mask, N=sensitivity,
                                                 pfa=np.sqrt(pfa), enl=enl, wi=wi, wo=wo)
            hv_outliers = kdistribution.detector(image[1, ...], mask=mask, N=sensitivity,
                                                 pfa=np.sqrt(pfa), enl=enl, wi=wi, wo=wo)
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = kdistribution.detector(image[0, ...], mask=mask, N=sensitivity,
                                                 pfa=1 - np.sqrt(1 - pfa), enl=enl, wi=wi, wo=wo)
            hv_outliers = kdistribution.detector(image[1, ...], mask=mask, N=sensitivity,
                                                 pfa=1 - np.sqrt(1 - pfa), enl=enl, wi=wi, wo=wo)
            outliers = hh_outliers | hv_outliers

    elif detector == 'wishart':
        image = utils.db2in(image)
        outliers = wishart.detector(image, mask=mask, pfa=pfa, enl=enl, wi=wi, wo=wo)

    elif detector == 'nis':
        image = utils.db2in(image)
        nis_transform = normsum.transform(image, mask=mask, wi=wi, wo=wo)
        nis_enl = utils.calc_enl(np.where(nis_transform < np.nanmedian(nis_transform) * 2, nis_transform, np.nan))
        warnings.filterwarnings('ignore') # supress warnings for the gamma detector - all checks already made in normsum.transform
        outliers = gamma.detector(nis_transform, mask=mask, pfa=pfa, enl=nis_enl, wi=wi, wo=wo)

    elif detector == 'idpolrad':
        image = utils.db2in(image)

        outliers = dpolrad.detector(image, mask=mask, pfa=pfa, test_window=wi, train_window=wo)

    else:
        warnings.warn(f'Detector not recognized. Use one of the following: \
                      \'gamma\', \'lognorm\', \'k\', \'wishart\', \'nis\', \'idpolrad\'. \
                      You gave the following detector: {detector}', category=UserWarning)
        return 0

    return utils.remove_small_objects(outliers, minsize)
