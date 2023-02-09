
cfar-object-detection :ice_cube: :mag:
======
Iceberg detection in SAR imagery using CFAR filters.
The code is tailored for 40m pixel spacing SAR imagery.
Detectors tested on dual-band ALOS-2 and Sentinel-1 imagery
 
## Overview
Includes 5 different CFAR detectors and contrast enhancement techniques for iceberg detection:
* Wishart (K. Conradsen 2003)
* Log-normal (D. J. Crisp 2004)
* Gamma (D. J. Crisp 2004)
* K-Distribution (Wesche and Dierking 2012; Brekke 2009)
* iDPolRad (A. Marino 2016)
* Normalized Intensity Sum (C. Liu 2015)


## Usage

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
        'sensitivity' : 40 # Only used by the k-algorithm. Higher number means slower and more precise
        }

# run detection
gamma_and_outliers = cfar.detector.run(db_image, mask, **init)
init.update({'method': 'OR'})
gamma_or_outliers = cfar.detector.run(sen1_image, mask, **init)

```

## References
C. Oliver and S. Quegan, Understanding synthetic aperture radar images, vol. 53, no. 9. 1997.

C. Liu, “A Dual-Polarization Ship Detection Algorithm,” Def. Res. Dev. Canada, no. November, 2015.

C. Brekke, "Automatic ship detection based on satellite SAR", Tech. Rep. May, Norwegian Defence Research Establishment (FFI), 2009.

C. Wesche and W. Dierking, “Iceberg signatures and detection in SAR images in two test regions of the Weddell Sea, Antarctica,” J. Glaciol., vol. 58, no. 208, pp. 325-339, 2012, doi: 10.3189/2012J0G11J020.

D. J. Crisp, “The State-of-the-Art in Ship detection in Synthetic Aperture Radar Imagery,” 2004.

K. Conradsen, A. A. Nielsen, J. Schou, and H. Skriver, “A test statistic in the complex wishart distribution and its application to change detection in polarimetric SAR data,” IEEE Trans. Geosci. Remote Sens., vol. 41, no. 1, pp. 4–19, 2003, doi: 10.1109/TGRS.2002.808066.

A. Marino, W. Dierking, and C. Wesche, “A Depolarization Ratio Anomaly Detector to Identify Icebergs in Sea Ice Using Dual-Polarization SAR Images,” IEEE Trans. Geosci. Remote Sens., vol. 54, no. 9, pp. 5602–5615, 2016, doi: 10.1109/TGRS.2016.2569450.

I. H. Soldal, W. Dierking, A. A. Korosov, and A. Marino, “Automatic detection of small icebergs in fast ice using satellite Wide-Swath SAR images,” Remote Sens., vol. 11, no. 7, pp. 1-24, 2019, doi: 10.3390/rs11070806.