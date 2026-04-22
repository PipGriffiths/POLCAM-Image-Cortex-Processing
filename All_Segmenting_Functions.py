#The Complete Functions that can be used for segmenting
#This contains, segmenting_tool.py, local_thresholding.py

'''
No longer used.
'''
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

def subtract_bkgnd(image, offset):
    image = image.astype(np.float32)
    bkgnd_corrected_img = image - offset
    final = np.clip(bkgnd_corrected_img, 0, 65535).astype(np.uint16)
    return final

def divide_bkgnd(image, factor):
    image = image.astype(np.float32)
    bkgnd_corrected_img = image / factor
    final = bkgnd_corrected_img.astype(np.uint16)
    return final


# RGB, S0, AoLP, DoLP Function
def Polcam(picture, polariser_string):
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

# Napari Semi-Automatic Functions
def follow_path(image, p1, p2):
    grad = sobel(image)
    cost = 1 / (grad + 1e-6)
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    path, _ = route_through_array(cost, p1, p2, fully_connected=True)
    return np.array(path)

def build_mask(image, points):
    paths = []
    points = np.array(points).astype(int)
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        segment = follow_path(image, p1, p2)
        paths.append(segment[:-1]) 
    contour = np.vstack(paths)
    mask = polygon2mask(image.shape, contour)
    mask = binary_fill_holes(mask)
    return mask, contour

def Segment_SemiAuto_Napari(image):
    """
    Opens Napari, allows user to segment, and returns the mask/contour.
    Includes a manual event loop to work in Spyder/Jupyter.
    """
    
    # 1. Local result container
    results = {'mask': None, 'contour': None}

    # 2. Initialize Viewer
    viewer = napari.Viewer()
    viewer.add_image(image, name='cell')
    points_layer = viewer.add_points(ndim=image.ndim, name='clicks', size=5, face_color='red')
    points_layer.mode = 'add'
    # 3. Define the Callback
    def on_enter(viewer):
        points_data = points_layer.data
        
        if len(points_data) < 3:
            print(f"Need at least 3 points. Currently: {len(points_data)}")
            return

        print("Calculating segmentation...")
        
        try:
            # Run Logic
            layer_data = viewer.layers['cell'].data
            mask, contour = build_mask(layer_data, points_data)
            
            # Update results
            results['mask'] = mask
            results['contour'] = contour
            
            print("Segmentation saved internally.")
            print("Closing Napari...")
            viewer.close()
            
        except Exception as e:
            print(f"Error: {e}")

    viewer.bind_key("Enter", on_enter)
    print(">>> Please segment the image and press ENTER <<<")
    
    # 4. Manual Blocking Loop
    # This loop forces Python to wait until the window is closed.
    # processEvents() ensures the GUI doesn't freeze while waiting.
    
    app = QApplication.instance()
    
    if app is None:
        # If no application instance exists (rare in interactive environments)
        # we can't safely proceed without starting a full Qt loop, which we want to avoid.
        raise RuntimeError(
            "Could not find Qt Application instance. Ensure you are running "
            "in an environment (like Spyder or Jupyter) where the Qt backend is enabled."
        )

    # Loop while the results haven't been captured AND the viewer window is still visible.
    while results['mask'] is None and viewer.window.qt_viewer.isVisible():
        # Process GUI events (clicks, draws, etc.) to keep the window responsive.
        app.processEvents() 
        # Sleep briefly to save CPU.
        time.sleep(0.05)
    
    # 5. Check for manual closure before returning
    if results['mask'] is None:
        print("Viewer closed manually without segmentation.")
    
    contour = results['contour']
    contour = contour[:, [1, 0]] # swap y,x coordinates to x,y on the contour
    
    # 5. Return captured results
    return results['mask'], contour

# Full Automatic Function - local thresholding
def Segment_Auto_Local(image):
    '''
    Parameters
    ----------
    image : 2D image.

    Returns
    -------
    dilated_mask : Mask to unwrap.
    binary_local : Binary filled mask.
    '''
    
    # colors, AoLP, DoLP, S0 = Polcam(image, '[-45 0; 90 45]')
    # image = S0
    normalised_image = (image - image.min()) / (image.max() - image.min())
    
    image = ski.util.img_as_ubyte(normalised_image)
    
    block_size = 251
    
    local_thresh = ski.filters.threshold_local(image, block_size, offset = 5)
    binary_local = image > local_thresh
    binary_local = ski_seg.clear_border(binary_local)
    
    cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = 250)
    filled_mask = binary_fill_holes(cleared_mask)
    
    selem = ski_morph.disk(1) 
    
    for i in range(13): #up to interpretation
        filled_mask = ski_morph.erosion(filled_mask, selem)
    
    largest_edge = find_boundaries(filled_mask, mode='inner')
    dilated_mask = morphology.dilation(largest_edge, selem)
    
    for  i in range(5): #up to interpretation
        dilated_mask = morphology.dilation(dilated_mask, selem)
    return dilated_mask, filled_mask


def MajorMinor_Axes(mask, scale):
    region = measure.regionprops(measure.label(mask))[0]
    major_axis = region.major_axis_length * scale
    minor_axis = region.minor_axis_length * scale
    return major_axis, minor_axis


def Segment_Auto_Global(image, top_percentage_pixels):
    percentile_thresh = np.percentile(image, 100 - top_percentage_pixels)  
    binary = image > percentile_thresh
    
    segment_edge = find_boundaries(binary, mode = 'outer')
    labels = measure.label(segment_edge)
    
    cleaned = morphology.remove_small_objects(labels, min_size = 500)
    largest_edge = (cleaned == np.argmax(np.bincount(cleaned.flat)[1:]) + 1)

    filled_mask = binary_fill_holes(largest_edge)
    label_image = measure.label(filled_mask)
    
    return label_image

#Unwrapping Function
def Unwrapping_1_Pixel(image, dilated_mask):
    contours = measure.find_contours(dilated_mask, 0.5)
    ring = max(contours, key = len)
    start_idx = np.argmin(ring[:, 0])
    ring = np.roll(ring, -start_idx, axis=0)
    x,y = ring
    values_in_ring = map_coordinates(image, [x,y], order = 1)
    values_norm = (values_in_ring - values_in_ring.min()) / (values_in_ring.max() - values_in_ring.min())
    return values_norm

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


def Unwrap_NotNorm(image, mask, thickness, smoothing):
    
    contours = measure.find_contours(mask, 0.5)
    
    if not contours:
        print("No contour found.")
        return np.zeros((10, 10))
    
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
    
    num_points = len(contour_final)
    
    half_thickness = thickness // 2
    d_indices = np.arange(-half_thickness, thickness - half_thickness)
    s_indices = np.arange(num_points)
    
    S, D_grid = np.meshgrid(s_indices, d_indices) # Shape (thickness, perimeter)
    
    sample_y = contour_final[S, 0] + ny[S] * D_grid
    sample_x = contour_final[S, 1] + nx[S] * D_grid

    coords = np.vstack((sample_y.flatten(), sample_x.flatten()))
    
    unwrapped_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0)
    unwrapped_image = unwrapped_flat.reshape((thickness, num_points))
    
    return unwrapped_image


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

def BinAveraging(image, bin_size):
    '''
    For every bin - find the max and divide by the max! Use this to normalise every bin but shapes will be consitent with the width of the cortex!
    This can take FWHM
    This FWHM is a local parameter
    
    Now describe a different parameter for the FWHM for the global using an average of all the FWHMs

    '''
    '''
    Go through and take bin of a specified pixel length
    Then slice it for the column, should end up with a lot of gaussian like profiles
    In a for loop, then average all the slices in the bin for a single gaussian
    '''
    height, width = image.shape
    num_bins = width // bin_size
    
    profiles = []
    
    for i in range(num_bins):
        start_x = i * bin_size
        end_x = start_x + bin_size
        
        current_bin = image[:, start_x : end_x]
        
        avg_profile = np.mean(current_bin, axis = 1)
        avg_profile = np.flip(avg_profile)
        profiles.append(avg_profile)
    
    return profiles


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

def ThreshThick(image, block, it, is_remainder=False):
    if image.size == 0: return np.zeros_like(image, dtype=bool)

    local_thresh = ski.filters.threshold_local(image, block, 
                                               method='gaussian', offset=5)
    binary_local = image > local_thresh
    
    # Only apply boundary padding if this is the "remainder" noise-heavy segment
    if is_remainder:
        binary_local[:, :2] = False
        binary_local[:, -2:] = False

    # Standard clean-up
    binary_local[image == 0] = False
    cleared_mask = ski_morph.remove_small_objects(binary_local, min_size=500)
    
    selem = ski_morph.disk(1) 
    for i in range(it):
        cleared_mask = ski_morph.erosion(cleared_mask, selem)
    
    clean_mask = ski_morph.remove_small_objects(cleared_mask, min_size=200)
    
    for i in range(it):
        clean_mask = ski_morph.dilation(clean_mask, selem)
    
    return binary_fill_holes(clean_mask)

def counter(image, bin_size):
    cortex_widths = []
    num_bins = len(image[0]) // bin_size
    for i in range(num_bins):
        points = np.where(image[:,i] == 1)
        cortex_widths.append(len(points[0]))
    return cortex_widths

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

def SplitByType(image):
    rows, cols = image.shape
    col_intensities = np.mean(image, axis=0)
    
    def get_indices(center, width=100):
        # Generates a range and handles wrap-around
        return np.arange(center - width, center + width) % cols

    # 1. Identify first cortex
    peak1 = np.argmax(col_intensities)
    idx1 = get_indices(peak1)
    idx1_sorted = np.sort(idx1)
    
    # 2. Mask and identify second cortex
    masked_intensities = col_intensities.copy()
    masked_intensities[idx1] = -1 
    peak2 = np.argmax(masked_intensities)
    idx2 = get_indices(peak2)
    idx2_sorted = np.sort(idx2)
    
    # 3. Define the Remainder (The "Gaps")
    # We find every index NOT used by the two cortex slices
    used_indices = np.unique(np.concatenate([idx1, idx2]))
    remainder_mask = np.ones(cols, dtype=bool)
    remainder_mask[used_indices] = False
    idx3 = np.where(remainder_mask)[0]
    idx3_sorted = np.sort(idx3)
    
    # Extract images based on these indices
    img1 = image[:, idx1_sorted]
    img2 = image[:, idx2_sorted]
    img3 = image[:, idx3_sorted]
    
    # Return as tuples (data, indices)
    return [(img1, idx1_sorted), (img2, idx2_sorted), (img3, idx3_sorted)]

def xticks(coords, step = 50):
    tick_indices = np.arange(0, len(coords), step)
    tick_labels = coords[tick_indices]
    return [tick_indices, tick_labels]

def CombineThresholded(thresholded_results, original_shape):
    """
    Recombines mask segments. Pastes the background first, 
    then overlays the cortex slices.
    """
    # Create blank canvas
    full_reconstruction = np.zeros(original_shape, dtype=np.uint8)
    
    # Sort results so the 'Remainder' (usually the 3rd item) is processed first
    # We can do this by checking the length of the coordinate array
    # The 'Remainder' usually has many more indices than the +/- 250px slices
    sorted_results = sorted(thresholded_results, key=lambda x: len(x[1]), reverse=True)
    
    for mask, coords in sorted_results:
        # Paste mask into the original global column positions
        full_reconstruction[:, coords] = mask
        
    return full_reconstruction

def CortexSim(width, height, cortex_thickness, 
                                 thickness_randomness, blur_sigma):
    img = np.zeros((height, width))
    centerline = np.zeros(width)
    current_y = height // 2
    
    reversion_strength = 0.015 
    random_step_size = 1.5     
    
    for i in range(1, width):
        pull = (height // 2 - current_y) * reversion_strength
        current_y += pull + np.random.normal(0, random_step_size)
        centerline[i] = current_y

    centerline = gaussian_filter(centerline, sigma=15)
    
    brightness_profile = gaussian_filter(np.random.uniform(0.7, 1.0, width), sigma=30)

    for i in range(width):### instead set thickness based on a gaussian function not random uniform
        local_thick = cortex_thickness + np.random.uniform(-thickness_randomness, thickness_randomness)
        y_indices = np.arange(height) #needs to have furrows
        dist = np.abs(y_indices - centerline[i])
        column_intensity = np.exp(-(dist**2) / (2 * (local_thick/3)**2))
        
        img[:, i] = column_intensity * brightness_profile[i]

    img = gaussian_filter(img, sigma=blur_sigma)
    noise = np.random.normal(0, 0.05, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return (img * 255).astype(np.uint8)

def RedefineCentre(image):
    rows, cols = image.shape
    maxlist = np.max(image, axis = 0)
    startx = np.argmin(maxlist)
    piece1 = image[:, startx:]
    piece2 = image[:, :startx]
    newimage = np.concatenate((piece1, piece2), axis = 1)
    return newimage





