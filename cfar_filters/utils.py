import functools
import numpy as np
import pandas as pd
from skimage import morphology
from skimage.transform import downscale_local_mean, resize

def db2in(image):
    """
    Convert decibel to intensity

    Parameters:
    ----------
    image : numpy.ndarray (N,X,Y)
        SAR image in decibel

    Returns:
    ----------
    image : numpy.ndarray (N,X,Y)
        SAR image in intensity
    """
    return np.power(10, np.divide(image, 10))


def in2db(image):
    """
    Convert intensity to decibel

    Parameters:
    ----------
    image : numpy.ndarray (N,X,Y)
        SAR image in intensity

    Returns:
    ----------
    image : numpy.ndarray (N,X,Y)
        SAR image in decibel

    """
    return 10 * np.log10(np.absolute(image))


def calc_enl(samples):
    """
    simple mom method for estimating the ENL
    theoretically only works for gamma, but are commonly used for other distributions as well

    Parameters:
    ----------
    samples : numpy.ndarray (N)
        single dimensional numpy array containing the samples used for ENL estimation

    Returns:
    ----------
    enl : float
        ENL estimate

    """
    return np.nanmean(samples)**2 / np.nanstd(samples)**2

# TODO: this function should use the mask and only calculate statistics within the valid area.
def smells_like(image, K=1000):
    """
    Often, SAR images are visualized in decibel scale.
    However, processing is often executed in linear intensity scale.
    This leads to errors, if eg. a decibel image is given as an input to a function expecting linear intensity.
    This function executes a 'soft' test in order to try to guess what format the input is.
    The function is used to warn about likely wrong formats.

    Parameters:
    ----------
    image : numpy.ndarray (N,X,Y)
        SAR image
    K : integer
        Number of samples used to create the statistics

    Returns:
    ----------
    smells_like : string
        'decibel'/'intensity'/'fishy'
    """

    if len(image.shape) == 2:
        samples = pd.DataFrame(image[np.random.choice(image.shape[0], K), np.random.choice(image.shape[1], K)].T)
    else:
        samples = pd.DataFrame(image[:, np.random.choice(image.shape[1], K), np.random.choice(image.shape[2], K)].T)

    describe = samples.describe()

    # if mean < 0 and std > 1 and min < -20 => smells like db
    if np.all([describe.loc['mean'] < 0, describe.loc['std'] > 1, describe.loc['min'] < -20]):
        smells_like = 'decibel'
    # if 0 < mean < 1 and 0 < std < 1 and min > 0 => smells like intensity
    elif np.all([0 < describe.loc['mean'], describe.loc['mean'] < 1, 0 < describe.loc['std'],
                describe.loc['std'] < 1, describe.loc['min'] > 0]):
        smells_like = 'intensity'
    else:
        smells_like = 'fishy'

    return smells_like


def remove_small_objects(image, N):
    """
    Remove small binary objects from a binary image

    Parameters:
    ----------
    image : numpy.ndarray (X,Y)
        Image to be masked
    N : integer
        minimum object size

    Returns:
    ----------
    image : numpy.ndarray (X,Y)
        cleaned image

    """
    out = morphology.remove_small_objects(image, N)
    return out


def mask_edges(image, N, fill=False):
    """
    Masks the edges of an image

    Parameters:
    ----------
    image : numpy.ndarray (X,Y)
        Image to be masked
    fill : fill value
        e.g. False or np.nan

    Returns:
    ----------
    image : numpy.ndarray (X,Y)
        masked image

    """
    image[0:N, :] = fill
    image[:, 0:N] = fill
    image[-N:, :] = fill
    image[:, -N:] = fill

    return image


def multiscale(func):
    """
    Multiscale processing decorator
    Apply binary image filters to several downsampled versions of the original image.
    Tailored for the CFAR detector format, i.e., cfar(image, mask, init)[outliers]
    
    Use as:
    @multiscale
    def multiscale_cfar(image, mask, init):
        return cfar.detector.run(image, mask, **init)
    
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        image = kwargs['image']
        mask = kwargs['mask']
        init = kwargs['init']
        levels = kwargs['levels']
        
        outliers = func(image, mask, init)
        
        N = 2**np.linspace(1, levels, levels).astype(int)

        for level in N:
            
            image_downsampled = in2db(downscale_local_mean(db2in(image), (1, level, level))) # downscale using the local mean in linear intensity
            mask_downsampled = downscale_local_mean(mask, (level, level))>0
            
            # apply detector to the downscaled image
            outliers_downsampled = func(image_downsampled, mask_downsampled, init)*1 # *1 for conversion to int8
            # upsample the results and merge with the previous iteration
            outliers_upsampled = resize(outliers_downsampled, outliers.shape, anti_aliasing=False, preserve_range=True)
            outliers = outliers + outliers_upsampled

        return outliers

    return wrapper
