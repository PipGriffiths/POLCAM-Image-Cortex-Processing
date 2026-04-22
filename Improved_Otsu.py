import napari
import numpy as np
from skimage.filters import threshold_otsu, gaussian
import skimage.io
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_closing, disk
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, map_coordinates

#%%

def Maskmaker(image, top_percentage_pixels):
    
    blurred_image = gaussian(image, sigma=2)
    
    thresh = threshold_otsu(blurred_image)
    binary = blurred_image > thresh
    
    closed_binary = binary_closing(binary, footprint=disk(5))
    filled_mask = binary_fill_holes(closed_binary)
    cleaned_mask = morphology.remove_small_objects(filled_mask, min_size=500)
    labels = measure.label(cleaned_mask)
    
    # Check if we found any objects
    if labels.max() == 0:
        print("No object found. Try adjusting sigma or threshold.")
        # Return empty/default values to prevent crash
        return None, None, None, np.array([])

    # Select the largest object
    # We look at the counts of each label (excluding 0/background) and pick the most frequent one
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    largest_mask = (labels == largest_label) 
    # 6. Generate the Edge for Analysis
    # Now we generate the edge from the solid mask, ensuring it is continuous.
    largest_edge = find_boundaries(largest_mask, mode='inner')

    # Global cell parameters
    # We calculate properties based on the solid mask, not just the edge
    region = measure.regionprops(measure.label(largest_mask))[0]
    major_axis = region.major_axis_length
    minor_axis = region.minor_axis_length

    # --- Existing contour analysis logic ---
    contours = measure.find_contours(largest_mask, 0.5) # Use mask, not edge, for cleaner contours
    
    if not contours:
        return major_axis, minor_axis, largest_edge, np.array([])

    ring = max(contours, key=len)
    start_idx = np.argmin(ring[:, 0])
    ring = np.roll(ring, -start_idx, axis=0)
    y, x = ring.T
    
    # Sample the original image using the coordinates
    values_in_ring = map_coordinates(image, [y, x], order=1)
    
    # Safety check for zero division if image is flat (rare)
    if values_in_ring.max() - values_in_ring.min() == 0:
         values_norm = values_in_ring
    else:
        values_norm = (values_in_ring - values_in_ring.min()) / (values_in_ring.max() - values_in_ring.min())
    
    return major_axis, minor_axis, largest_edge, values_norm