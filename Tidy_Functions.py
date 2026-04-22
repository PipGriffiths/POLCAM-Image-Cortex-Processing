'''
Tidy functions only!
'''
#%% Imports
import numpy as np
from napari_polcam._functions import PolarisationCameraImage
from matplotlib.colors import hsv_to_rgb
import napari
from skimage.filters import sobel
from skimage.graph import route_through_array
from skimage.draw import polygon2mask
from scipy.ndimage import binary_fill_holes
from qtpy.QtWidgets import QApplication
import time
import skimage as ski
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
import skimage.segmentation as ski_seg
import skimage.morphology as ski_morph
from skimage.morphology import binary_closing, disk
from scipy.ndimage import  map_coordinates, gaussian_filter1d
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
from scipy.spatial import distance_matrix
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

#%% Preprocessing (pre Polcam)

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

#%% POLCAM code

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


#%% Segmenting Functions (follow_path and build_mask)

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

#%% SemiAutomatic Segmenting

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


#%% Automatic (Local) Thresholding

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


#%% Cell properties

def CellProperties(mask):
    # 1. Load and clean
    mask = mask > 0.5

    coords = np.column_stack(np.where(mask))
    
    # 2. Long Axis (Feret Diameter)
    hull = ConvexHull(coords)
    hull_points = coords[hull.vertices]
    dists = distance_matrix(hull_points, hull_points)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    p1, p2 = hull_points[i], hull_points[j]
    
    # 3. Calculate Center and Perpendicular Direction
    midpoint = (p1 + p2) / 2
    direction = p2 - p1
    perp_dir = np.array([-direction[1], direction[0]])
    perp_dir = perp_dir / np.linalg.norm(perp_dir)

    # 4. "Grow" the short axis from the center until it hits the mask edge
    def get_edge_point(start_point, unit_vec, mask):
        current_point = start_point.copy()
        # Step outward until we hit the background or image edge
        while True:
            next_point = current_point + unit_vec
            r, c = int(round(next_point[0])), int(round(next_point[1]))
            
            # Check bounds and mask value
            if (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] 
                and mask[r, c]):
                current_point = next_point
            else:
                break
        return current_point

    # Find the two edge points along the perpendicular line
    s1 = get_edge_point(midpoint, perp_dir, mask)
    s2 = get_edge_point(midpoint, -perp_dir, mask)
    short_axis_len = np.linalg.norm(s1 - s2)

    # 5. Visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    
    # Long Axis (Red)
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r', linewidth=2, label=f'Long: {np.linalg.norm(p1-p2):.1f}px')
    
    # Short Axis (Blue) - Now constrained to mask boundary
    plt.plot([s1[1], s2[1]], [s1[0], s2[0]], 'b', linewidth=2, label=f'Short: {short_axis_len:.1f}px')
    
    plt.plot(midpoint[1], midpoint[0], 'go') # Center point
    plt.legend()
    plt.title("Constrained Axis Calculation")
    plt.show()

    return np.linalg.norm(p1-p2), short_axis_len

    
#%% Unwrapping

def Unwrap(image, mask, thickness, smoothing, normalising = False):
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
    
    if image.ndim == 3:    
        channels = []
        for i in range(image.shape[2]):
            sampled = map_coordinates(image[:,:,i], coords, order = 1, mode = 'constant', cval = 0)
            channels.append(sampled.reshape(thickness, num_points))
        unwrapped_image = np.stack(channels, axis = -1)
    else:        
        unwrapped_flat = map_coordinates(image, coords, order=1, mode='constant', cval=0)
        unwrapped_image = unwrapped_flat.reshape((thickness, num_points))
    
    if normalising:     
        low, high = unwrapped_image.min(), unwrapped_image.max()
        if high > low:
            unwrapped_image = (unwrapped_image - low) / (high - low)
    
    return unwrapped_image

#%% Redefine the centre

def RedefineCentre(unwrap, start = None):
    '''
    Use start = None ONLY for s0 unwrapping. Otherwise use the start point from the s0 unwrapping which you must do first

    Parameters
    ----------
    unwrap : TYPE
        DESCRIPTION.
    start : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if unwrap.ndim == 3:
        intensity_profile = np.max(np.mean(unwrap, axis = 2), axis = 0)
    else:
        intensity_profile = np.max(unwrap, axis = 0)

    if start == None:
        startx = np.argmin(intensity_profile)
    else:
        startx = start
    redefined = np.roll(unwrap, -startx, axis = 1)
    
    # piece1 = unwrap[:, startx:]
    # piece2 = unwrap[:, :startx]
    # newimage = np.concatenate((piece1, piece2), axis = 1)
    
    if start == None:    
        return redefined, startx
    else:
        return redefined

#%% Getting values along columns for a certain bin_size
def BinAveraging(image, bin_size):
    '''
    Go through and take bin of a specified pixel length
    Then slice it for the column, should end up with a lot of gaussian like profiles
    In a for loop, then average all the slices in the bin for a single gaussian
    
    Could add normalisation
    '''
    height, width = image.shape
    num_bins = width // bin_size
    
    profiles = []
    
    for i in range(num_bins):
        start_x = i * bin_size
        end_x = start_x + bin_size
        
        # if end_x <= width:
        #     current_bin = image[:, start_x : end_x]
        # else:
        #     # Wrap around logic: Concatenate end of image with start
        #     pixels_from_end = width - start_x
        #     pixels_from_start = bin_size - pixels_from_end
        #     current_bin = np.concatenate((image[:, start_x:], image[:, :pixels_from_start]), axis=1)
        
        current_bin = image[:, start_x : end_x]
        
        avg_profile = np.mean(current_bin, axis = 1)
        avg_profile = np.flip(avg_profile)
        profiles.append(avg_profile)
    
    return profiles



def BinAveraging_Zeroing(image, bin_size, zero_threshold = 3):
    """
    Bins the image, wraps around the edges, and enforces 
    row-level zero masking before averaging.
    """
    height, width = image.shape
    num_bins = width // bin_size
    
    profiles = []
    
    for i in range(num_bins):
        start_x = i * bin_size
        end_x = start_x + bin_size
        
        # 1. Handle selection with Wrapping
        if end_x <= width:
            current_bin = image[:, start_x : end_x]
        else:
            # Wrap around logic: Concatenate end of image with start
            pixels_from_end = width - start_x
            pixels_from_start = bin_size - pixels_from_end
            current_bin = np.concatenate((image[:, start_x:], image[:, :pixels_from_start]), axis=1)
            
        # 2. Row-level Zero Masking (Done BEFORE averaging)
        # Create a boolean mask: True for rows that contain at least one 0
        zero_counts = np.sum(current_bin == 0, axis=1)
        mask = (zero_counts >= zero_threshold)
        
        
        # Set the entire row to 0 in the current_bin
        # This replaces the values so they don't contribute to the average
        current_bin[mask] = 0
        
        # 3. Calculate Average
        # Now we calculate the mean as normal. 
        # Note: If a row was set to 0, the average will be 0.
        avg_profile = np.mean(current_bin, axis=1)
        
        # 4. Flip and store
        avg_profile = np.flip(avg_profile)
        profiles.append(avg_profile)
    
    return profiles






#%% Local Thresholding the unwrapped image for a mask

def ThreshThick(image, block, it, offset, is_remainder=False):
    
    if image.size == 0: return np.zeros_like(image, dtype=bool)

    local_thresh = ski.filters.threshold_local(image, block, 
                                               method='gaussian', offset=offset)
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
    
    clean_mask = ski_morph.remove_small_objects(cleared_mask, min_size=1000)
    
    # for i in range(it):
        # clean_mask = ski_morph.dilation(clean_mask, selem)

    return binary_fill_holes(clean_mask)

#%% Counting the number of 1 in a column for a speicified bin_size

def counter(image, bin_size):
    rows, cols = image.shape
    cortex_widths = []
    bincortex_widths = []
    num_bins = cols // bin_size
    cols = image.shape[1]
    
    for i in range(cols):
        points = np.where(image[:, i] == 1)
        cortex_widths.append(len(points[0]))
    
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size
        current_bin_widths = cortex_widths[start:end]
        bin_average = sum(current_bin_widths) / bin_size
        bincortex_widths.append(bin_average)
        
    return bincortex_widths

#%% Splitting up the unwrapped image to make thresholding easier and xticks for preserving location

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

def xticks(idxs, step = 50):
    tick_indices = np.arange(0, len(idxs), step)
    tick_labels = idxs[tick_indices]
    return [tick_indices, tick_labels]

#%% Recombining the Split images

def CombineThresholded(thresholded_results, original_shape):
    """
    Recombines mask segments. Pastes the background first, 
    then overlays the cortex slices.
    """
    full_reconstruction = np.zeros(original_shape, dtype=np.uint8)
    sorted_results = sorted(thresholded_results, key=lambda x: len(x[1]), reverse=True)
    
    for mask, coords in sorted_results:
        full_reconstruction[:, coords] = mask
        
    return full_reconstruction


#%% Full split, threshold and recombination
def UnwrappedThresholding(image, block, iterations):
    splitup = SplitByType(image)
    processed_list = []
    for i, (img, coords) in enumerate(splitup):
        # The 3rd item (index 2) is our remainder segment
        is_rem = (i == 2)
        mask = ThreshThick(img, block = 31, it = 2, is_remainder=is_rem)
        processed_list.append((mask, coords))
    final_full_mask = CombineThresholded(processed_list, image.shape)
    return final_full_mask

#%% Simulating some cortex data

def CortexSim(width, height, cortex_thickness, thickness_randomness, blur_sigma):
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


#%% Specific profile range

def ProfileSpecBin(image, start, end):
    '''
    Parameters
    ----------
    image : the unwrapped image.
    start : start column number.
    valuerange : end column number.

    You can pass lists into the start and end arguments and do lots at once

    Returns
    -------
    profiles : slices the column range inputted, does an average of every row, 
        makes a list of that and returns the list.
    '''
    if type(start) and type(end) == int:
        current_bin = image[:, start : end]
        avg_profile = np.mean(current_bin, axis = 1)
        avg_profile = np.flip(avg_profile)
        return avg_profile
    
    if type(start) and type(end) == list:
        profiles = []
        for i in range(len(start)):
            current_bin = image[:, start[i] : end[i]]
            avg_profile = np.mean(current_bin, axis = 1)
            avg_profile = np.flip(avg_profile)
            profiles.append(avg_profile)
        return profiles
    else:
        return print('Please give integers or lists for the start and end you want')

#%% Elbow peak finder for one line

def PeakFinder(x_data, y_data):
    start_pt = np.array([x_data[0], y_data[0]])
    end_pt = np.array([x_data[-1], y_data[-1]])
    
    line_vec = end_pt - start_pt
    line_len = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_len
    
    distances = []
    for i in range(len(x_data)):
        p = np.array([x_data[i], y_data[i]])
        p_rel = p - start_pt
        dist = np.linalg.norm(p_rel - np.dot(p_rel, line_unit_vec) * line_unit_vec)
        distances.append(dist)
    
    peak_idx = np.argmax(distances)
    peak_x = x_data[peak_idx]
    peak_y = y_data[peak_idx]
    
    return peak_x, peak_y


#%% Elbow PeakFinder for an image

def PeakFinderWholeImage(image):
    """
    Finds the (x, y) coordinates of the peak deviation 
    for every column in an image.
    """
    height, width = image.shape
    x_data = np.arange(height)
    peak_coords = []
    peak_idxs = []
    
    for col_idx in range(width):
        y_data = image[:, col_idx]
        
        # Define start and end points for this specific column
        start_pt = np.array([0, y_data[0]])
        end_pt = np.array([height - 1, y_data[-1]])
        
        # Calculate line properties
        line_vec = end_pt - start_pt
        line_len = np.linalg.norm(line_vec)
        
        # Handle case where start and end points are identical 
        # (prevents division by zero)
        if line_len == 0:
            peak_coords.append((0, y_data[0]))
            continue
            
        line_unit_vec = line_vec / line_len
        
        # Vectorized distance calculation for this column
        p_rel = np.stack([x_data, y_data], axis=1) - start_pt
        # Using the cross-product magnitude equivalent for perpendicular distance
        # dist = |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / sqrt((x2-x1)^2 + (y2-y1)^2)
        # However, keeping your projection method for consistency:
        projections = np.dot(p_rel, line_unit_vec)[:, np.newaxis] * line_unit_vec
        distances = np.linalg.norm(p_rel - projections, axis=1)
        
        # Find index of max distance
        peak_idx = np.argmax(distances)
        peak_coords.append((x_data[peak_idx], y_data[peak_idx]))
        peak_x = x_data[peak_idx]
        peak_idxs.append(peak_x)
        
    return peak_idxs


#%% Gauss and Fit

def GaussianiserAndFit(ydata):
    # 1. Setup indices
    ydata = np.asanyarray(ydata)
    x = np.arange(len(ydata))
    maxidx = PeakFinder(x, ydata)
    l = len(ydata)
    l4 = l // 10
    
    if maxidx < l4 or maxidx + l4 > l: 
        return np.zeros(len(ydata)) 

    # 2. Slice the data
    start, end = maxidx - l4, maxidx + l4
    x_slice = x[start:end]
    y_slice = ydata[start:end]

    # dist = np.abs(x_slice - maxidx)
    # weights = np.exp(-(dist**2) / (2 * (l4/15)**2))**4


    def gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    popt, _ = curve_fit(gauss, x_slice, y_slice, 
                        p0=[np.max(y_slice), maxidx, l4/4])#,sigma=1.0/weights)
    return gauss(x, *popt)


#%% Lorentz and Fit

def FitLorentzian(ydata, FWHM=False, trim=False):
    def lorentzian(x, a, x0, gamma):
        return a * (gamma**2 / ((x - x0)**2 + gamma**2))

    ydata = np.asanyarray(ydata)
    x = np.arange(len(ydata))

    max_idx, max_val = PeakFinder(x, ydata)    

    dist_from_center = np.abs(x - max_idx)
    weights = dist_from_center + 0.001

    p0 = [max_val, max_idx, 5.0]
    popt, _ = curve_fit(lorentzian, x, ydata, p0=p0, sigma = weights)
        
    fitdata = lorentzian(x, *popt)
    
    if FWHM:    
        return fitdata, 2 * np.abs(popt[2])
    else:
        return fitdata



#%% Fit (weighted) Gaussian

def FitGaussian(ydata, FWHM = False):
    def gauss(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    xdata = np.arange(len(ydata))
    max_idx, max_val = PeakFinder(xdata, ydata)
    target = 5 * max_val / 6
    
    left_side = ydata[:max_idx]
    right_side = ydata[max_idx:]
    
    idx_left = np.argmin(np.abs(left_side - target))
    idx_right = np.argmin(np.abs(right_side - target)) + max_idx
        
    guess = [np.max(ydata), np.argmax(ydata), idx_right - idx_left]
    
    dist_from_center = np.abs(xdata - np.argmax(ydata))
    weights = dist_from_center + 0.5
    
    popt,pcov = curve_fit(gauss, xdata, ydata, p0 = guess, sigma = weights)

    fitdata = gauss(xdata, *popt)
    if FWHM:
        FWHM = 2*np.sqrt(2*np.log(2))*np.abs(popt[2])
        return fitdata, FWHM
    return fitdata



#%%


def FitGeneralizedNormal(ydata, FWHM=False):
    # a: amplitude, x0: center, alpha: scale (width), beta: shape
    def gen_normal(x, a, x0, alpha, beta):
        return a * np.exp(-(np.abs(x - x0) / alpha) ** beta)
    
    xdata = np.arange(len(ydata))
    max_idx, max_val = PeakFinder(xdata, ydata)
    
    # Initial guess
    # beta=2.5 is a good starting point for a flat-topped peak
    guess = [np.max(ydata), max_idx, 5.0, 2.5] 
    
    popt, pcov = curve_fit(gen_normal, xdata, ydata, p0=guess)
    fitdata = gen_normal(xdata, *popt)
    
    if FWHM:
        # FWHM formula for Gen. Normal: 2 * alpha * (ln(2))^(1/beta)
        alpha, beta = popt[2], popt[3]
        fwhm_val = 2 * alpha * (np.log(2))**(1/beta)
        return fitdata, fwhm_val
    
    return fitdata





#%% findpeaks simple DEFUNCT
def findpeaks(unwrapped_image):
    maxvalues = np.max(unwrapped_image, axis = 0)
    heavysmoothing = gaussian_filter(maxvalues, sigma = 80)
    peaks, _ = find_peaks(heavysmoothing)
    return peaks

#%% interpolating normaliser
from scipy.interpolate import interp1d

def NormalisingWidth(data, target_length):

    data = np.asanyarray(data)
    current_length = data.shape[-1]
    old_x = np.linspace(0, 1, current_length)
    new_x = np.linspace(0, 1, target_length)
    f = interp1d(old_x, data, kind='cubic', axis=-1)
    #try kind = 'cubic' also
    
    return f(new_x)

#%% local thresholding more efficient

def LocalSegment(image, erosions, dilations, padding = False):
    
    if padding:
        pad_amount = 100
        image = np.pad(image, pad_width = pad_amount, mode = 'constant', constant_values = np.min(image))
        
    block_size = 251

    selem = ski_morph.disk(1) 
    local_thresh = ski.filters.threshold_local(image, block_size, offset = 0.005*np.max(image))
    binary_local = image > local_thresh
    binary_local = ski_seg.clear_border(binary_local)
    
    cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = image.size*0.002)
    sealed_mask = ski_morph.binary_closing(cleared_mask, footprint = ski_morph.disk(6))
    filled_mask = binary_fill_holes(sealed_mask)       
    filled_mask = ski_seg.clear_border(filled_mask)
    
    for i in range(erosions):
        filled_mask = ski_morph.erosion(filled_mask, selem)
        
    filled_mask = ski_morph.remove_small_objects(filled_mask, min_size = image.size*0.015)

    for i in range(dilations):
        filled_mask = morphology.dilation(filled_mask, selem)            

    if padding:
        cropped_image = filled_mask[100:-100, 100:-100]
        return cropped_image

    return filled_mask

#%% reader

def reader(path, imagetype):
    from pathlib import Path
    import skimage.io
    images = []
    files = Path(rf'{path}').rglob(rf'*.{imagetype}')
    for file in files:
        images.append(skimage.io.imread(file))
    return images

#%% figuresaver

def figuresaver(imagelist, foldername, typename, formatting = 'png', original = False):
    import io
    import matplotlib.pyplot as plt
    from zipfile import ZipFile
    
    if len(imagelist) == 1:
        content = imagelist[0]
        filename = f'{foldername}.{formatting}'
        
        if original:
            import shutil
            shutil.copy(content, filename)
        else:
            if isinstance(content, plt.Figure):
                content.savefig(filename, format = formatting)
                plt.close(content)
            else:
                fig, ax = plt.subplots()
                ax.imshow(content, cmap='gray')
                ax.set_title(f'{typename} 0')
                plt.savefig(filename, format=formatting)
                plt.close(fig)
        print('Done')
        return
    
    with ZipFile(f"{foldername}.zip", "w") as my_zip:
        
        if original:
            for i, content in enumerate(imagelist):    
                my_zip.write(content, arcname = f'{typename}_{i}.{formatting}')
        
        else:    
            for i, content in enumerate(imagelist):
                
                if isinstance(content, plt.Figure):
                    img_buffer = io.BytesIO()
                    content.savefig(img_buffer, format = 'png')
                    plt.close(content)
                else:    
                    fig, ax = plt.subplots()
                    ax.imshow(content, cmap = 'gray')
                    ax.set_title(f'{typename} {i}')
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format=formatting)
                    plt.close(fig)
                    
                # 4. Write to zip
                my_zip.writestr(f'{typename}_{i}.{formatting}', img_buffer.getvalue())
                
#%% PCA, FFT, dilated mask
from sklearn.decomposition import PCA
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def cell_pca(image, n_components):
    
    pca = PCA(n_components = n_components)
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)
    
    return reconstructed


def cell_fft(image, radius):
    
    transform = fft2(image)
    shift = fftshift(transform)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    
    mask = (x - ccol)**2 + (y - crow)**2 >= radius**2
    shift_filtered = shift * mask
    ishift = ifftshift(shift_filtered)
    img_back = ifft2(ishift)
    
    return np.abs(img_back)


def dilationmask(mask, number):
    
    selem = morphology.disk(1)
    largest_edge = find_boundaries(mask, mode='inner')
    dilated_mask = morphology.dilation(largest_edge, selem)

    for  i in range(number): #up to interpretation
        dilated_mask = morphology.dilation(dilated_mask, selem)
    
    return dilated_mask

#%% super res thresholding


def SuperThresholding(image, sigma, dilations, redilations):
    selem = ski_morph.disk(1)
    newimg = gaussian_filter(image, sigma)
    thresh = threshold_otsu(newimg)
    otsumask = newimg > thresh
    dilated_mask = morphology.dilation(otsumask, selem)
    
    for i in range(dilations):
        dilated_mask = morphology.dilation(dilated_mask, selem)
    
    filled_mask = binary_fill_holes(dilated_mask)
    
    for i in range(dilations):
        filled_mask = morphology.erosion(filled_mask, selem)
   
    filled_mask = ski_morph.remove_small_objects(filled_mask, min_size = 0.01*image.size)
    
    largest_edge = find_boundaries(filled_mask, mode = 'inner')
    
    for i in range(redilations):
        largest_edge = morphology.dilation(largest_edge, selem)
    

    return largest_edge, filled_mask

#%% image histogram, cropper, set_top

def histo(image, bins, log = False):
    import matplolib.pyplot as plt
    if log:
        plt.hist(image.ravel(), bins = bins, log = True)
    if not log:
        plt.hist(image.ravel(), bins = bins)
    plt.title("Pixel Value Distribution")

def cropper(image, thresh = 4, pad = 100, normalising = False):
    mask = image > thresh
    rows, cols = image.shape
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis = 0)
    y1, x1 = coords.max(axis = 0)
    cropped = image[max(0, y0 - pad) : min(rows, y1 + pad), 
                             max(0, x0 - pad) : min(cols, x1 + pad)]
    if normalising:
            cropped = (cropped - cropped.min()) / (cropped.max() - cropped.min())
    
    return cropped

def set_top(image, top, zeroes = False):
    
    if np.max(image) != 1:
        image = (image - image.min()) / (image.max() - image.min())
    if not zeroes:
        threshold = np.percentile(image, top)
        image[image >= threshold] = 1
    if zeroes:
        image[image != 0] = 1        
    return image

#%% SigmoidLorentzian

def SigLor(x_data, y_data):

    def lorentzian(x, amp, cen, gamma):
        return amp / (1 + ((x - cen) / gamma)**2)

    def left_side_step(x, L, x0, k):
        return L / (1 + np.exp(k * (x - x0)))

    def combined_model(x, l_amp, l_cen, l_gamma, s_L, s_x0, s_k):
        return lorentzian(x, l_amp, l_cen, l_gamma) + left_side_step(x, s_L, s_x0, s_k)
    
    # Check if y_data is a list of datasets or a single dataset
    # (Using the logic from your original snippet)
    if len(np.shape(y_data)) > 1:
        
        l_fits = []
        s_fits = []
        fwhms = []
        
        for i in range(len(y_data)):
            ydata = y_data[i]
            
            start_buffer = len(x_data) // 10
            peak_idx = np.argmax(ydata[start_buffer:]) + start_buffer
            peak_x = x_data[peak_idx]
            peak_y = ydata[peak_idx]
            
            plateau_len = int(len(x_data) * 0.15)
            plateau_y = np.mean(ydata[:plateau_len])
            
            step_back = len(x_data) * 0.15
            max_step_x = peak_x - (len(x_data) * 0.08) 
            
            # [l_amp, l_cen, l_gamma, s_L, s_x0, s_k]
            # Note: For Lorentzian, gamma is roughly half the width. 
            # I've kept your sigma-based initialization as a starting point for gamma.
            p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]

            low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
            high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
            
            weights = np.ones(len(x_data))
            window = int(len(x_data) * 0.1) 
            weights[max(0, peak_idx-window) : min(len(x_data), peak_idx+window)] = 10.0
            
            popt, _ = curve_fit(combined_model, x_data, ydata, 
                                p0=p0, bounds=(low, high), sigma=1/weights)
            
            l_fit = lorentzian(x_data, *popt[:3])
            l_fits.append(l_fit)
            s_fit = left_side_step(x_data, *popt[3:])
            s_fits.append(s_fit)
            
            # For Lorentzian, FWHM is simply 2 * gamma
            FWHM = 2 * popt[2]
            fwhms.append(FWHM)
            
        return l_fits, s_fits, fwhms
    
    else:
        # Single dataset logic
        start_buffer = len(x_data) // 10
        peak_idx = np.argmax(y_data[start_buffer:]) + start_buffer
        peak_x = x_data[peak_idx]
        peak_y = y_data[peak_idx]
        
        plateau_len = int(len(x_data) * 0.15)
        plateau_y = np.mean(y_data[:plateau_len])
        
        step_back = len(x_data) * 0.15
        max_step_x = peak_x - (len(x_data) * 0.08) 
        
        p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]
    
        low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
        high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
        
        weights = np.ones(len(x_data))
        window = int(len(x_data) * 0.1) 
        weights[max(0, peak_idx-window) : min(len(x_data), peak_idx+window)] = 10.0
        
        popt, _ = curve_fit(combined_model, x_data, y_data, 
                            p0=p0, bounds=(low, high), sigma=1/weights)
        
        l_fit = lorentzian(x_data, *popt[:3])
        s_fit = left_side_step(x_data, *popt[3:])
        FWHM = 2 * popt[2]
        
        return l_fit, s_fit, FWHM


#%% SigmoidGaussian

def SigGauss(x_data, y_data):

    def gaussian(x, amp, cen, sigma):
        return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

    def left_side_step(x, L, x0, k):
        return L / (1 + np.exp(k * (x - x0)))

    def combined_model(x, g_amp, g_cen, g_sigma, s_L, s_x0, s_k):
        return gaussian(x, g_amp, g_cen, g_sigma) + left_side_step(x, s_L, s_x0, s_k)
    
    # Check if y_data is a list of datasets or a single dataset
    if len(np.shape(y_data)) > 1:
        
        g_fits = [] # Renamed from l_fits
        s_fits = []
        fwhms = []
        
        for i in range(len(y_data)):
            ydata = y_data[i]
            
            start_buffer = len(x_data) // 10
            peak_idx = np.argmax(ydata[start_buffer:]) + start_buffer
            peak_x = x_data[peak_idx]
            peak_y = ydata[peak_idx]
            
            plateau_len = int(len(x_data) * 0.15)
            plateau_y = np.mean(ydata[:plateau_len])
            
            step_back = len(x_data) * 0.15
            max_step_x = peak_x - (len(x_data) * 0.08) 
            
            # [g_amp, g_cen, g_sigma, s_L, s_x0, s_k]
            p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]

            low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
            high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
            
            weights = np.ones(len(x_data))
            window = int(len(x_data) * 0.1) 
            weights[max(0, peak_idx-window) : min(len(x_data), peak_idx+window)] = 10.0
            
            popt, _ = curve_fit(combined_model, x_data, ydata, 
                                p0=p0, bounds=(low, high), sigma=1/weights)
            
            g_fit = gaussian(x_data, *popt[:3])
            g_fits.append(g_fit)
            s_fit = left_side_step(x_data, *popt[3:])
            s_fits.append(s_fit)
            
            # For Gaussian, FWHM is approx 2.355 * sigma
            FWHM = 2.355 * popt[2]
            fwhms.append(FWHM)
        print('bing bong')
        return g_fits, s_fits, fwhms
    
    else:
        # Single dataset logic
        start_buffer = len(x_data) // 10
        peak_idx = np.argmax(y_data[start_buffer:]) + start_buffer
        peak_x = x_data[peak_idx]
        peak_y = y_data[peak_idx]
        
        plateau_len = int(len(x_data) * 0.15)
        plateau_y = np.mean(y_data[:plateau_len])
        
        step_back = len(x_data) * 0.15
        max_step_x = peak_x - (len(x_data) * 0.08) 
        
        p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]
    
        low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
        high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
        
        weights = np.ones(len(x_data))
        window = int(len(x_data) * 0.1) 
        weights[max(0, peak_idx-window) : min(len(x_data), peak_idx+window)] = 10.0
        
        popt, _ = curve_fit(combined_model, x_data, y_data, 
                            p0=p0, bounds=(low, high), sigma=1/weights)
        
        g_fit = gaussian(x_data, *popt[:3])
        s_fit = left_side_step(x_data, *popt[3:])
        # FWHM update
        FWHM = 2.355 * popt[2]
        
        return g_fit, s_fit, FWHM


#%% maxcoordinate finders - update so that we use elbow point!!!

def maxcoord_saver(image):
    rowindices = np.argmax(image, axis = 0)
    colindices = np.arange(image.shape[1])
    coordinates = np.stack((rowindices, colindices), axis=-1)
    tuplelist = [tuple(coord) for coord in coordinates]
    return tuplelist
    
#%% otsu threshold of an fft cell

def Otsu(image):
    fft = cell_fft(image = image, radius = 5)
    thresh = threshold_otsu(fft)
    binary = image > thresh
    cleared_mask = ski_morph.remove_small_objects(binary, min_size = 250)
    filled_mask = binary_fill_holes(cleared_mask)
    return filled_mask

#%% padding normaliser


def Normaliser(data, target_width):
    data = np.asanyarray(data)
    current_width = data.shape[-1] # Works for both 1D and 2D
    
    # If it's already the right size (or bigger), return as is
    if current_width >= target_width:
        return data
    
    # Calculate how much padding is needed
    pad_size = target_width - current_width
    
    # Create the 'padding' block of NaNs
    # This matches the 'height' of your data automatically
    padding_shape = list(data.shape)
    padding_shape[-1] = pad_size
    nan_padding = np.full(padding_shape, np.nan)
    
    # Stick them together horizontally
    return np.concatenate([data, nan_padding], axis=-1)
    
#%% dynamic ROI finder

def Window(image, val_min=0, val_max=6000, target_min=1, target_max=50):
    """
    Centers a dynamic window on the brightest pixel of each column.
    Strictly clips boundaries to 0 and 49 (no wrap-around).
    """
    rows, cols = image.shape
    output = np.zeros_like(image)
    
    # Scaling constants
    val_range = val_max - val_min
    target_range = target_max - target_min

    for i in range(cols):
        column_data = image[:, i]
        
        # 1. Find the brightest pixel index and its value
        peak_row = np.argmax(column_data)
        peak_value = column_data[peak_row]
        
        # 2. Calculate window size based on brightness
        clamped_peak = max(val_min, min(val_max, peak_value))
        window_size = target_min + (clamped_peak - val_min) * (target_range / val_range)
        window_size = int(round(window_size))
        
        # 3. Calculate centering
        half_window = window_size // 2
        start = peak_row - half_window
        end = start + window_size
        
        # 4. Strict Boundary Clipping (The No-Wrap Logic)
        # If start is -5, it becomes 0. If end is 55, it becomes 50.
        actual_start = max(0, start)
        actual_end = min(rows, end)
        
        # 5. Extract the slice
        # Because 'output' is zeros, the areas outside actual_start/end 
        # are automatically "padded" with 0.
        output[actual_start:actual_end, i] = column_data[actual_start:actual_end]
        
    return output
    
#%% aggressive ROI finder


def PowerWindow(image, val_min=0, val_max=6000, target_min=1, target_max=50, power=0.75):
    rows, cols = image.shape
    output = np.zeros_like(image)
    
    target_range = target_max - target_min
    val_range = val_max - val_min

    peaks = PeakFinderWholeImage(image)

    for i in range(cols):
        column_data = image[:, i]
        peak_row = peaks[i]
        peak_value = column_data[peak_row]
        
        # 1. Normalize the peak value to a 0.0 - 1.0 scale
        clamped_peak = max(val_min, min(val_max, peak_value))
        normalized_val = (clamped_peak - val_min) / val_range
        
        # 2. Apply the Power Law (0.75)
        # This determines how "fast" the window grows
        scaled_factor = normalized_val ** power
        
        # 3. Map back to the 1-50 range
        window_size = target_min + (scaled_factor * target_range)
        window_size = int(round(window_size))
        
        # 4. Centering and Boundary Clipping (No Wrap)
        half_window = window_size // 2
        start = max(0, peak_row - half_window)
        end = min(rows, start + window_size)
        
        # 5. Extract to the blank canvas
        output[start:end, i] = column_data[start:end]
        
    return output
    
