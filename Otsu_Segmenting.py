import numpy as np
from skimage.filters import threshold_otsu
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes, map_coordinates


def maskmaker(image, top_percentage_pixels):
    
    #Set the threshold
    thresh = threshold_otsu(image)
    percentile_thresh = np.percentile(image, 100 - top_percentage_pixels)  
    binary = image > percentile_thresh
    
    #Getting a nice clean edge
    segment_edge = find_boundaries(binary, mode = 'outer')
    labels = measure.label(segment_edge)
    
    cleaned = morphology.remove_small_objects(labels, min_size = 500)
    largest_edge = (cleaned == np.argmax(np.bincount(cleaned.flat)[1:]) + 1)

    # Global cell parameters
    filled_mask = binary_fill_holes(largest_edge)
    label_image = measure.label(filled_mask)

    region = measure.regionprops(label_image)[0]
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length

    # This part could be copied and adapted to include AoLP and DoLP and even rgb?
    contours = measure.find_contours(largest_edge, 0.5)
    ring = max(contours, key = len)
    start_idx = np.argmin(ring[:, 0])
    ring = np.roll(ring, -start_idx, axis=0)
    y,x = ring.T
    values_in_ring = map_coordinates(image, [y,x], order = 1)
    values_norm = (values_in_ring - values_in_ring.min()) / (values_in_ring.max() - values_in_ring.min())
    
    return major_axis, minor_axis, largest_edge, values_norm