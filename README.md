
cfar-object-detection :ice_cube: :mag:
======
Iceberg detection in SAR imagery using CFAR filters.
The code is tailored for 40m pixel spacing SAR imagery.
Detectors tested on dual-band ALOS-2 and Sentinel-1 imagery
 
## Overview
Includes 4 different CFAR detectors:
* WISHART (K. Conradsen 2003)
* LogNormal (D. J. Crisp 2004)
* Gamma (D. J. Crisp 2004)
* K-Distribution (C. Liu 2018)

And 2 different detection transformations:
* DPolRad (A. Marino 2016)
* Normalized Intensity Sum (C. Liu 2015)

TODO:
* Add examples

## Usage

```
import cfar_filters as cfar

image = cfar.utils.db2in(decibel_image) # make sure to convert image to intensity

# detectors
wishart_outliers = cfar.wishart.detector(image, mask=mask, pfa=pfa, enl=enl)

lognorm_hh_outliers = cfar.lognormal.detector(in2db(image[0,...]), mask=mask, pfa=pfa)
lognorm_hv_outliers = cfar.lognormal.detector(in2db(image[1,...]), mask=mask, pfa=pfa)
lognorm_outliers = lognorm_hh_outliers&lognorm_hv_outliers

gamma_hh_outliers = cfar.gamma.detector(image[0,...], mask=mask, pfa=pfa, enl=enl)
gamma_hv_outliers = cfar.gamma.detector(image[1,...], mask=mask, pfa=pfa, enl=enl)
gamma_outliers = gamma_hh_outliers&gamma_hv_outliers

k_hh_outliers = cfar.kdistribution.detector(image[0,...], mask=mask, N=block_size, offset=False, pfa=pfa, enl=enl)
k_hv_outliers = cfar.kdistribution.detector(image[1,...], mask=mask, N=block_size, offset=False, pfa=pfa, enl=enl)
k_outliers = k_hh_outliers&k_hv_outliers
```

## References
C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.

C. Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

C. Liu, “Method for Fitting K-Distributed Probability Density Function to Ocean Pixels in Dual-Polarization SAR,” Can. J. Remote Sens., vol. 44, no. 4, pp. 299–310, 2018, doi: 10.1080/07038992.2018.1491789.

D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.

K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver, “A test statistic in the complex wishart distribution and its application to change detection in polarimetric SAR data,” IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4–19, 2003, doi: 10.1109/TGRS.2002.808066.

A. Marino, W. Dierking, and C. Wesche, “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602–5615, 2016, doi: 10.1109/TGRS.2016.2569450.
