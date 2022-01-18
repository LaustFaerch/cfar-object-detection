
cfar-object-detection :ice_cube: :droplet: :mag:
======

## Overview
Includes 3 different CFAR detectors:
* WISHART (Conradsen 2003)
* LogNormal (Crisp 2004)
* K-Distribution (Chen Liu 2018)

And 2 different detection transformations:
* DPolRad (Armando Marino 2018)
* Normalized Intensity Sum (Chen Liu 2015)

TODO:
* get warnings under control
* write documentation
* Add examples
* K-distribution sometimes fails at the image borders - maybe due to low amount of samples?

## Usage

## Citations

## References
C. Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.
C. Liu, “Method for Fitting K-Distributed Probability Density Function to Ocean Pixels in Dual-Polarization SAR,” Can. J. Remote Sens., vol. 44, no. 4, pp. 299–310, 2018, doi: 10.1080/07038992.2018.1491789.
C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.
D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.
K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver, “A test statistic in the complex wishart distribution and its application to change detection in polarimetric SAR data,” IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4–19, 2003, doi: 10.1109/TGRS.2002.808066.
