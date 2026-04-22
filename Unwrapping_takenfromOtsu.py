import numpy as np
from skimage import measure
from scipy.ndimage import map_coordinates

def Unwrapper(image, mask):
    '''
    Parameters
    ----------
    image : 2D image.
    mask : Also 2D image of True and False.

    Returns
    -------
    values_norm : Brightness on the segmented coordinates unwrapped.
    '''

    contours = measure.find_contours(mask, 0.5) # Use mask, not edge, for cleaner contours    
    ring = max(contours, key=len)
    start_idx = np.argmin(ring[:, 0])
    ring = np.roll(ring, -start_idx, axis=0)
    x, y = ring.T
    
    # Sample the original image using the coordinates
    values_in_ring = map_coordinates(image, [y, x], order=1)
    
    # Safety check for zero division if image is flat (rare)
    if values_in_ring.max() - values_in_ring.min() == 0:
         values_norm = values_in_ring
    else:
        values_norm = (values_in_ring - values_in_ring.min()) / (values_in_ring.max() - values_in_ring.min())
    
    return values_norm