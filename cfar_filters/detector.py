"""
CFAR Detection on 40-meter SAR data.

If using this code, please use the following citation:
Færch, L., Dierking, W., Hughes, N., and Doulgeris, A. P.: 
A Comparison of CFAR Object Detection Algorithms for Iceberg Identification in L- and C-band SAR Imagery of the Labrador Sea, 
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

from . import kdistribution, lognormal, wishart, normsum, dpolrad, gamma, utils

def run(image, mask, detector='gamma', method='AND', pfa=1e-9, enl=10.7, minsize=2, sensitivity=40):

    if detector == 'gamma':
        image = utils.db2in(image)

        if method == 'AND':
            hh_outliers = gamma.detector(image[0, ...], mask=mask, pfa=np.sqrt(pfa), enl=enl)
            hv_outliers = gamma.detector(image[1, ...], mask=mask, pfa=np.sqrt(pfa), enl=enl)
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = gamma.detector(image[0, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), enl=enl)
            hv_outliers = gamma.detector(image[1, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa), enl=enl)
            outliers = hh_outliers | hv_outliers

    elif detector == 'lognorm':

        if method == 'AND':
            hh_outliers = lognormal.detector(image[0, ...], mask=mask, pfa=np.sqrt(pfa))
            hv_outliers = lognormal.detector(image[1, ...], mask=mask, pfa=np.sqrt(pfa))
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = lognormal.detector(image[0, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa))
            hv_outliers = lognormal.detector(image[1, ...], mask=mask, pfa=1 - np.sqrt(1 - pfa))
            outliers = hh_outliers | hv_outliers

    elif detector == 'k':
        image = utils.db2in(image)

        if method == 'AND':
            hh_outliers = kdistribution.detector(image[0, ...], mask=mask, N=sensitivity,
                                                 pfa=np.sqrt(pfa), enl=enl)
            hv_outliers = kdistribution.detector(image[1, ...], mask=mask, N=sensitivity,
                                                 pfa=np.sqrt(pfa), enl=enl)
            outliers = hh_outliers & hv_outliers

        elif method == 'OR':
            hh_outliers = kdistribution.detector(image[0, ...], mask=mask, N=sensitivity,
                                                 pfa=1 - np.sqrt(1 - pfa), enl=enl)
            hv_outliers = kdistribution.detector(image[1, ...], mask=mask, N=sensitivity,
                                                 pfa=1 - np.sqrt(1 - pfa), enl=enl)
            outliers = hh_outliers | hv_outliers

    elif detector == 'wishart':
        image = utils.db2in(image)
        outliers = wishart.detector(image, mask=mask, pfa=pfa, enl=enl)

    elif detector == 'nis':
        image = utils.db2in(image)
        nis_transform = normsum.transform(image, mask=mask)
        nis_enl = utils.calc_enl(np.where(nis_transform < np.nanmedian(nis_transform) * 2, nis_transform, np.nan))
        outliers = gamma.detector(nis_transform, mask=mask, pfa=pfa, enl=nis_enl)

    elif detector == 'idpolrad':
        image = utils.db2in(image)
        outliers = dpolrad.detector(image, mask=mask, pfa=pfa)

    else:
        warnings.warn(f'Detector not recognized. Use one of the following: \'gamma\', \'lognorm\', \'k\', \'wishart\', \'nis\', \'idpolrad\'. You gave the following detector: {detector}', category=UserWarning)
        return 0

    return utils.remove_small_objects(outliers, minsize)
