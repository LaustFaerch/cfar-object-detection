import numpy as np

from . import kdistribution, lognormal, wishart, normsum, dpolrad, gamma, utils

def detect(image, mask, detector='gamma', method='AND', pfa=1e-9, enl=10.7, minsize=2, sensitivity=40):

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
        return 0

    return utils.remove_small_objects(outliers, minsize)
