
cfar-object-detection :ice_cube: :mag:
======
Constant False Alarm Rate (CFAR) detection for SAR images.
Developed for detecting icebergs in Sentinel-1 and ALOS-2 imagery.
Implemented using Numba for fast parallel convolutions.
 
## Overview
Includes 5 different CFAR detectors and contrast enhancement techniques for iceberg detection:
* Wishart (K. Conradsen 2003)
* Log-normal (D. J. Crisp 2004)
* Gamma (D. J. Crisp 2004)
* K-Distribution (Wesche and Dierking 2012; Brekke 2009)
* iDPolRad (A. Marino 2016)
* Normalized Intensity Sum (C. Liu 2015)

The implementation have been tested on Sentinel-1 EW and IW, ALOS-2 Wide Beam, and ICEYE Stripmap images.


## Usage

For CFAR detection on single-band images use the individual detectors:

```Python
import rasterio as rio
import cfar_filters as cfar

# load your SAR image as a numpy array
# image format should be a numpy.ndarray(float32) of shape (2,X,Y) with the order: HH, HV
image_filename = Path('sentinel1-image.tif')
with rio.open(image_filename) as src:
    in_image = src.read()

# define mask - could also include landmask
mask = np.nansum(in_image, axis=0)!=0

pfa = 1e-15  # probability of false alarm rate
enl = 10.7  # equivalent number of looks of the image
wi = 9  # diameter of the guard area
wo = 15  # diameter of the clutter estimation area

# using the gamma detector
gamma_outliers = cfar.gamma.detector(in_image, mask, pfa, enl, wi, wo)

# using the lognormal detector
db_image = cfar.utils.in2db(in_image)  # convert image to decibel using .utils
lognormal_outliers = cfar.lognormal.detector(db_image, mask, pfa, wi, wo)

# using the k detector
N = 40  # number of steps for the LUT
k_outliers = cfar.kdistribution.detector(in_image, mask, N, pfa, enl, wi, wo)

# remove objects smaller than 3 using .utils
gamma_outliers = cfar.utils.remove_small_objects(gamma_outliers, 2)

```

For CFAR detection on dual-band images use the cfar.detector.run:


```Python
import rasterio as rio
import cfar_filters as cfar

# load your SAR image as a numpy array
# image format should be a numpy.ndarray(float32) of shape (2,X,Y) with the order: HH, HV
image_filename = Path('sentinel1-image.tif')
with rio.open(image_filename) as src:
    in_image = src.read()

# define mask - could also include landmask
mask = np.nansum(in_image, axis=0)!=0

# convert between intensity/decibel using .utils
db_image = cfar.utils.in2db(in_image)

# initialize detector setting
init = {
        'detector' : 'gamma', # use: 'gamma', 'lognorm', 'k', 'wishart', 'nis', 'idpolrad'
        'method' : 'AND', # use: 'AND', 'OR'. Not used by 'wishart', 'nis', and 'idpolrad'
        'pfa' :  1e-15, # recommended range: 1e-21 - 1e-3
        'enl' : 10.7, # use sensor ENL, 10.7 for Sentinel-1 EW
        'minsize' : 2, # objects smaller than this size are removed
        'sensitivity' : 40, # Only used by the k-algorithm. Higher number means slower and more precise
        'wi' : 9 , # diameter of the guard area
        'wo' : 15  # diameter of the clutter estimation area
        }

# run detection (image should be in decibel)
gamma_and_outliers = cfar.detector.run(db_image, mask, **init)
init.update({'method': 'OR'})
gamma_or_outliers = cfar.detector.run(db_image, mask, **init)

```

The CFAR clutter estimation windows are round and hollow. Similar to the implementation in this paper, with the exception that the target area is always a single pixel: https://www.researchgate.net/figure/Schematic-of-the-target-guard-and-background-areas-used-in-the-traditional-CFAR_fig1_350936854

The wi and wo parameters control the diameter of the guard area, and the clutter estimation area. E.g., setting wi to 5 and wo to 10 would result in the following CFAR window:

```Python
        |-----------------------------wo-----------------------------|   ___
array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   |
       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],   |
       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],   wi
       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],   |
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  _|_
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, X, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
       [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

With the center pixel 'X' being compared against all pixels marked with '1', and '0' being the guard area.
The maximum supported diameter is 20. If wo is set to a value larger than 20, the CFAR window will be cropped to a square of size 41x41. This is to restrict computational cost as the filter calculations is implemented using Numba.

NB: The code will use all available CPU power. Keep this in mind when your CPU goes to 100%. 

## References
C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.

C. Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

C. Brekke, "Automatic ship detection based on satellite SAR", Tech. Rep. May, Norwegian Defence Research Establishment (FFI), 2009.

C. Wesche and W. Dierking, “Iceberg signatures and detection in SAR images in two test regions of the Weddell Sea, Antarctica,” J. Glaciol., vol. 58, no. 208, pp. 325-339, 2012, doi: 10.3189/2012J0G11J020.

D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.

K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver, “A test statistic in the complex wishart distribution and its application to change detection in polarimetric SAR data,” IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4–19, 2003, doi: 10.1109/TGRS.2002.808066.

A. Marino, W. Dierking, and C. Wesche, “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602–5615, 2016, doi: 10.1109/TGRS.2016.2569450.

I. H. Soldal, W. Dierking, A. A. Korosov, and A. Marino, “Automatic detection of small icebergs in fast ice using satellite Wide-Swath SAR images,” Remote Sens., vol. 11, no. 7, pp. 1-24, 2019, doi: 10.3390/rs11070806.

## Cite

If using this, please use the following citation:

Færch, L., Dierking, W., Hughes, N., and Doulgeris, A. P.: A Comparison of CFAR Object Detection Algorithms for Iceberg Identification in L- and C-band SAR Imagery of the Labrador Sea, The Cryosphere Discuss. [preprint], https://doi.org/10.5194/tc-2023-17, in review, 2023. 

## License
The material is made available under the GNU General Public License v3.0: Copyright 2023, Laust Færch, of CIRFA - The Arctic University of Norway, UiT. All rights reserved.