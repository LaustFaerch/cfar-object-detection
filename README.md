
cfar-object-detection :ice_cube: :mag:
======
Iceberg detection in SAR imagery using CFAR filters.
The code is tailored for 40m pixel spacing SAR imagery.
Detectors tested on dual-band ALOS-2 and Sentinel-1 imagery
 
## Overview
Includes 5 different CFAR detectors and contrast enhancement techniques for iceberg detection:
* WISHART (K. Conradsen 2003)
* LogNormal (D. J. Crisp 2004)
* Gamma (D. J. Crisp 2004)
* K-Distribution (C. Liu 2018)
* DPolRad (A. Marino 2016)
* Normalized Intensity Sum (C. Liu 2015)


## Usage

```Python
import cfar_filters as cfar

# load your SAR image as a numpy array

# convert between intensity/decibel using cfar.utils:
image = cfar.utils.db2in(decibel_image)

# detectors need to be tuned with a desired pfa-level and ENL
pfa = 1e-9
enl = 10  # Sentinel-1 EW ENL is around 10

# defining a mask will make execution faster. Mask is boolean
mask = (np.nansum(image, axis=0)!=0)

# there are 3 single-channel detectors cfar.lognormal.detector(), cfar.gamma.detector(), and cfar.kdistribution.detector()
# single-channel detectors must be applied to individual bands (i.e., on HH / HV) and then the results must be combined later
# if combining the channels using boolean &, then the pfa must be adjusted accordingly
gamma_hh_outliers = cfar.gamma.detector(image[0,...], mask=mask, pfa=np.sqrt(pfa), enl=enl)
gamma_hv_outliers = cfar.gamma.detector(image[1,...], mask=mask, pfa=np.sqrt(pfa), enl=enl)

# merge the channels using boolean logic
gamma_outliers = gamma_hh_outliers&gamma_hv_outliers

# Wishart and idpolrad detectors is dual-channel
wishart_outliers = cfar.wishart.detector(image, mask=mask, pfa=pfa, enl=enl)
idpolrad_outliers = cfar.dpolrad.detector(image, mask=mask, pfa=pfa)

# transformations can be applied to transform dual-channel to single-channel
normsum_transform = cfar.normsum.transform(image, mask=mask)

# before applying a detector, we need to estimate the enl of the transformed image
# below, we only estimate enl using the background clytter (i.e., pixels with intensity below 2xmedian)
nis_enl = cfar.utils.calc_enl(np.where(normsum_transform<np.nanmedian(normsum_transform)*2, normsum_transform, np.nan))

# we can now apply a detector to the transformed image
nis_outliers = cfar.gamma.detector(normsum_transform, mask=mask, pfa=pfa, enl=nis_enl)


```

## References
C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.

C. Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

C. Liu, “Method for Fitting K-Distributed Probability Density Function to Ocean Pixels in Dual-Polarization SAR,” Can. J. Remote Sens., vol. 44, no. 4, pp. 299–310, 2018, doi: 10.1080/07038992.2018.1491789.

D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.

K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver, “A test statistic in the complex wishart distribution and its application to change detection in polarimetric SAR data,” IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4–19, 2003, doi: 10.1109/TGRS.2002.808066.

A. Marino, W. Dierking, and C. Wesche, “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602–5615, 2016, doi: 10.1109/TGRS.2016.2569450.

I. H. Soldal, W. Dierking, A. A. Korosov, and A. Marino, “Automatic detection of small icebergs in fast ice using satellite Wide-Swath SAR images,” Remote Sens., vol. 11, no. 7, pp. 1-24, 2019, doi: 10.3390/rs11070806.