import numpy as np
from napari_polcam._functions import PolarisationCameraImage
from matplotlib.colors import hsv_to_rgb

def HSV_maker(picture, polariser_string):
    """
    Parameters
    ----------
    picture : skimage.io read image.
    polariser_string : string in the form '[A B; C D]'
    offset_value : background level

    Returns
    -------
    color_images : rgb pretty picture
    AoLP (radians)
    DoLP ([0,1] range)

    """
    polcam_processor = PolarisationCameraImage(
        img = picture,
        method = 'Cubic spline interpolation',
        polariser_unit = polariser_string,
        offset = 0)
    
    channeled_data = polcam_processor.convert_unprocessed()

    I0 = channeled_data[0].astype(np.double)
    I45 = channeled_data[1].astype(np.double)
    I90 = channeled_data[2].astype(np.double)
    I135 = channeled_data[3].astype(np.double)

    #stokes parameters
    S0 = (I0 + I45 + I90 + I135)/2
    S1 = I0 - I90
    S2 = I45 - I135

    # for when you want to view the stokes images, you need these in the contrast limits
    max_s0 = np.max(S0)
    min_s0 = np.min(S0)
    dolp_min = 0.0
    dolp_max = 1.0

    AoLP = (1/2)*np.arctan2(S2,S1)
    DoLP = np.sqrt((S1**2 + S2**2)/(S0**2))

    hue = AoLP
    sat = DoLP
    value = S0

    hue = (hue + (np.pi/2)) / np.pi
    sat = (sat - dolp_min) / (dolp_max - dolp_min)
    value = (value - min_s0) / (max_s0 - min_s0) 

    hue[hue < 0] = 0
    sat[sat < 0] = 0
    value[value < 0] = 0
    hue[hue > 1] = 1
    sat[sat > 1] = 1
    value[value > 1] = 1

    numDim = len(hue.shape)
    hsv = np.stack([hue, sat, value], numDim)
    rgb = hsv_to_rgb(hsv)
    rgb = rgb*255
    HSVmap = rgb.astype(np.uint8)
    color_images = HSVmap
    
    return color_images, AoLP, DoLP, S0


