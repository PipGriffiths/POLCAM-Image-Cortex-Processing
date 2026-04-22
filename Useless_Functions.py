'''
Useless Functions
'''
#%% Imports
import numpy as np
from napari_polcam._functions import PolarisationCameraImage
from matplotlib.colors import hsv_to_rgb
import napari
from skimage.filters import sobel, threshold_otsu, gaussian
from skimage.graph import route_through_array
from skimage.draw import polygon2mask
from scipy.ndimage import binary_fill_holes
from qtpy.QtWidgets import QApplication
import time
import skimage as ski
import skimage.io
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
import skimage.segmentation as ski_seg
import skimage.morphology as ski_morph
from skimage.morphology import binary_closing, disk
from scipy.ndimage import  map_coordinates, gaussian_filter1d
from scipy.ndimage import gaussian_filter


#%% Global Thresholding (useless)

def Segment_Auto_Global(image, top):
     
    percentile_thresh = np.percentile(image, 100 - top)  
    binary = image > percentile_thresh
    
    segment_edge = find_boundaries(binary, mode = 'outer')
    labels = measure.label(segment_edge)
    
    cleaned = morphology.remove_small_objects(labels, min_size = 500)
    largest_edge = (cleaned == np.argmax(np.bincount(cleaned.flat)[1:]) + 1)
    
    selem = ski_morph.disk(1) 
    
    dilated_mask = morphology.dilation(largest_edge, selem)
    
    for j in range(5): #up to interpretation
        dilated_mask = morphology.dilation(dilated_mask, selem)

    filled_mask = binary_fill_holes(dilated_mask)
    
        
    for k in range(13): #up to interpretation
        filled_mask = ski_morph.erosion(filled_mask, selem)
            
    label_image = measure.label(filled_mask)
        
    return label_image

#%% Unwrapping Function 1 pixel
def Unwrapping_1_Pixel(image, dilated_mask):
    contours = measure.find_contours(dilated_mask, 0.5)
    ring = max(contours, key = len)
    start_idx = np.argmin(ring[:, 0])
    ring = np.roll(ring, -start_idx, axis=0)
    x,y = ring
    values_in_ring = map_coordinates(image, [x,y], order = 1)
    values_norm = (values_in_ring - values_in_ring.min()) / (values_in_ring.max() - values_in_ring.min())
    return values_norm

#%% Unwrapping with coords

def Unwrap_General_Contour(image, mask, thickness, smoothing):
    """
    measure.find_contours works by scanning from top left to bottom right for the first pixel that fits 0.5
        This cannot be changed by the user
    
    Unwraps an arbitrary closed shape (like a dividing cell) by sampling 
    inwards along the normal vectors of the contour.
    
    Parameters
    ----------
    image : 2D array
        The intensity image.
    mask : 2D array (bool or int)
        The binary mask of the cell object.
    thickness : int
        How many pixels deep (into the cell) you want to unwrap.
    smoothing : float
        Sigma for Gaussian smoothing of the contour. Crucial to prevent
        jagged/noisy sampling lines.
        
    Returns
    -------
    unwrapped : 2D array
        The straightened strip of the cortex.
    """
    
    # 1. Get the contour
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        print("No contour found.")
        return np.zeros((10, 10))
    
    # Take the longest contour (assumes it's the cell boundary)
    contour = max(contours, key=len)
    
    # 2. Smooth the contour
    # Raw pixel contours are jagged. We smooth them to get clean normal vectors.
    # We treat x and y coordinates as signals and smooth them.
    contour_smooth = gaussian_filter1d(contour, sigma=smoothing, axis=0)
    
    # Close the loop for smooth derivatives (wrap around)
    contour_smooth = np.vstack([contour_smooth, contour_smooth[0:2]])
    
    # 3. Calculate Derivatives (Tangents and Normals)
    # Gradient gives us (dy, dx) change along the path
    derivs = np.gradient(contour_smooth, axis=0)
    dy = derivs[:, 0]
    dx = derivs[:, 1]
    
    # Calculate Normals. 
    # For a counter-clockwise contour, (-dy, dx) points Inward, (dy, -dx) points Outward.
    # scikit-image contours are usually counter-clockwise.
    # We normalize them to length 1.
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Avoid division by zero
    magnitude[magnitude == 0] = 1 
    
    # Define Normal vectors (pointing Inward)
    # Note: If your image comes out "flipped" (looking outside), swap the signs below.
    nx = -dy / magnitude
    ny = dx / magnitude
    
    # Remove the extra point we added for closing the loop
    nx = nx[:-1]
    ny = ny[:-1]
    contour_final = contour_smooth[:-1]
    
    # 4. Generate the Sampling Grid
    num_points = len(contour_final)
    
    # Create arrays for the result
    half_thickness = thickness // 2
    d_indices = np.arange(-half_thickness, thickness - half_thickness)
    # We want a strip of width = perimeter, height = thickness
    # s = distance along contour, d = depth into cell
    s_indices = np.arange(num_points)
    
    # Create 2D mesh of indices
    S, D_grid = np.meshgrid(s_indices, d_indices) # Shape (thickness, perimeter)
    
    # Calculate sampling coordinates
    # P_sample = P_boundary + Normal * Depth
    # Remember contour is (row, col) -> (y, x)
    sample_y = contour_final[S, 0] + ny[S] * D_grid
    sample_x = contour_final[S, 1] + nx[S] * D_grid

    # 5. Map Coordinates
    # Stack for map_coordinates: (2, N_pixels)
    coords = np.vstack((sample_y.flatten(), sample_x.flatten()))
    
    unwrapped_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0)
    unwrapped_image = unwrapped_flat.reshape((thickness, num_points))
    
    #6. Find extrema to mark on both original and unwrapped images
    rows, cols = unwrapped_image.shape
    contour_row_idx = rows // 2
    contour_profile = unwrapped_image[contour_row_idx, :]
    
    max_s = np.argmax(contour_profile)
    min_s = np.argmin(contour_profile)
    
    max_d = contour_row_idx
    min_d = contour_row_idx
    
    unwrapped_coords = np.array([[max_s, min_s], [max_d, min_d]])
    
    max_flat_idx = max_d * cols + max_s
    min_flat_idx = min_d * cols + min_s
    
    bright_coords_original = (sample_x.flatten()[max_flat_idx], sample_y.flatten()[max_flat_idx])
    dim_coords_original = (sample_x.flatten()[min_flat_idx], sample_y.flatten()[min_flat_idx])
    
    original_coords = np.array([
        [bright_coords_original[0], dim_coords_original[0]], # X-coords
        [bright_coords_original[1], dim_coords_original[1]]  # Y-coords
    ])
    
    # Normalize output
    if unwrapped_image.max() > unwrapped_image.min():
        unwrapped_image = (unwrapped_image - unwrapped_image.min()) / (unwrapped_image.max() - unwrapped_image.min())
        
    return unwrapped_image, original_coords, unwrapped_coords

#%% Unwrapping with lower sampling (aiming to reduce jaggedness)
def Unwrap_Lower_Sampling(image, mask, thickness, smoothing, n):
    
    contours = measure.find_contours(mask, 0.5)
    contour = max(contours, key=len)
    contour_smooth = gaussian_filter1d(contour, sigma=smoothing, axis=0)
    contour_smooth = np.vstack([contour_smooth, contour_smooth[0:2]])
    
    derivs = np.gradient(contour_smooth, axis=0)
    dy = derivs[:, 0]
    dx = derivs[:, 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    magnitude[magnitude == 0] = 1 
    nx = -dy / magnitude
    ny = dx / magnitude
    nx = nx[:-1]
    ny = ny[:-1]
    contour_final = contour_smooth[:-1]
    
    s_indices_original = np.arange(len(contour_final))
    s_indices = s_indices_original[::n]

    # 4. Generate the Sampling Grid
    num_points = len(s_indices)    
    half_thickness = thickness // 2
    d_indices = np.arange(-half_thickness, thickness - half_thickness)

    # Create 2D mesh of indices
    S, D_grid = np.meshgrid(s_indices, d_indices) # Shape (thickness, perimeter)
    
    # Calculate sampling coordinates
    # P_sample = P_boundary + Normal * Depth
    # Remember contour is (row, col) -> (y, x)
    sample_y = contour_final[S, 0] + ny[S] * D_grid
    sample_x = contour_final[S, 1] + nx[S] * D_grid

    # 5. Map Coordinates
    # Stack for map_coordinates: (2, N_pixels)
    coords = np.vstack((sample_y.flatten(), sample_x.flatten()))
    
    unwrapped_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0)
    unwrapped_image = unwrapped_flat.reshape((thickness, num_points))
    
    # Normalize output
    if unwrapped_image.max() > unwrapped_image.min():
        unwrapped_image = (unwrapped_image - unwrapped_image.min()) / (unwrapped_image.max() - unwrapped_image.min())
        
    return unwrapped_image

#%% Finding the size of a furrow
def BrightWidth(image, bright_percentile):
    '''
    Go through an image and find the width in pixels of a bright region
    return a bin size of the bright region
    '''
    flat_data = image.flatten()
    
    threshold_percentile = 100.0 - bright_percentile
    brightness_threshold = np.percentile(flat_data, threshold_percentile)
    
    bright_mask = image > brightness_threshold
    
    columns_with_bright_pixels = np.where(np.any(bright_mask, axis=0))[0]
    
    if columns_with_bright_pixels.size == 0:
        print("No pixels found above the specified brightness threshold.")
        return 0
    
    min_x = columns_with_bright_pixels.min()
    max_x = columns_with_bright_pixels.max()
    
    width = max_x - min_x + 1
    coords = [min_x, max_x]
    
    return width, coords


#%% Good block size optimiser for local thresholding

def Adaptive_Local_Segment(s0, erosion, dilation):
    
    image = s0
    normalised_image = (image - image.min()) / (image.max() - image.min())
    
    image = ski.util.img_as_ubyte(normalised_image)
    
    best_block = None
    
    for block in range(51, 275, 10): 
        local_thresh = ski.filters.threshold_local(image, block, offset = 5)
        binary_local = image > local_thresh
        binary_local = ski_seg.clear_border(binary_local)
        cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = 250)
        filled_mask = binary_fill_holes(cleared_mask)
        
        if np.count_nonzero(filled_mask) > 2000:
            best_block = block
            break

    local_thresh = ski.filters.threshold_local(image, best_block, offset = 5)
    binary_local = image > local_thresh
    binary_local = ski_seg.clear_border(binary_local)
    cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = 250)
    filled_mask = binary_fill_holes(cleared_mask)
    
    
    selem = ski_morph.disk(1) 
    
    for i in range(erosion): #up to interpretation
        filled_mask = ski_morph.erosion(filled_mask, selem)
    
    largest_edge = find_boundaries(filled_mask, mode='inner')
    dilated_mask = morphology.dilation(largest_edge, selem)
    
    for  i in range(dilation): #up to interpretation
        dilated_mask = morphology.dilation(dilated_mask, selem)
    
    return dilated_mask, filled_mask, best_block


#%% Half-width half-maximum idea

def HWHM(line):
    '''
    Parameters
    ----------
    line : 1D numpy array.
    Returns: 2* the the half-width half-maximum
    TYPE
        DESCRIPTION.

    '''
    maximum = np.argmax(line)
    linehalf = line[maximum:]
    halfmax = np.max(linehalf) / 2
    halved = (np.abs(linehalf - halfmax)).argmin()
    return halved*2

#%%

