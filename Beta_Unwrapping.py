import numpy as np
from scipy.ndimage import map_coordinates


def Unwrapper_from_contour(image, contour_coords, width=20, smoothing_sigma=3):
    '''
    Unwraps a region around a given contour.

    Parameters
    ----------
    image : 2D image
    contour_coords : N x 2 array of (row, col) coordinates defining the contour.
                     (i.e., (y, x) coordinates).
    width : Integer. The half-width (in pixels) of the strip to unwrap
            perpendicular to the contour. Total strip width will be 2*width + 1.
    smoothing_sigma : Float. The standard deviation for Gaussian smoothing 
                      applied to the contour coordinates before calculating normals.

    Returns
    -------
    values_norm : Normalized brightness values along the smoothed contour.
    unwrapped_image : A 2D array (image) of the unwrapped region.
    smoothed_ring : The N x 2 array of smoothed contour coordinates.
    '''
    from scipy.ndimage import gaussian_filter1d
    ring = np.array(contour_coords)
    x, y = ring.T
    
    # Apply 1D Gaussian smoothing along the length of the contour (axis=0)
    x_smooth = gaussian_filter1d(x, smoothing_sigma, axis=0, mode='wrap')
    y_smooth = gaussian_filter1d(y, smoothing_sigma, axis=0, mode='wrap')
    
    # Use the smoothed coordinates for all subsequent steps
    x = x_smooth
    y = y_smooth

    smoothed_ring = np.vstack([x, y])
    num_points = len(ring) # Still the same number of points

    # 2. Calculate the Normalized Contour Values (Using Smoothed Coordinates)
    
    # Sample the original image using the *smoothed* contour coordinates
    values_in_ring = map_coordinates(image, [x, y], order=1)

    # Normalize the values
    v_min, v_max = values_in_ring.min(), values_in_ring.max()
    if v_max - v_min == 0:
        values_norm = values_in_ring
    else:
        values_norm = (values_in_ring - v_min) / (v_max - v_min)

    # 3. Calculate Perpendicular Coordinates for Unwrapping (Same Logic)
    
    # Calculate the tangent vector (delta_x, delta_y) at each point
    # We use np.roll to get the difference between the next and previous points.
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)

    # The normal vector (perpendicular to the tangent) is (-dy, dx)
    nx = -dy
    ny = dx
    
    # Normalize the normal vector to unit length
    norm_length = np.sqrt(nx**2 + ny**2)
    norm_length[norm_length == 0] = 1 # Avoid division by zero
    
    nx_unit = nx / norm_length
    ny_unit = ny / norm_length

    # Create the coordinate grid for the unwrapped image
    d = np.arange(-width, width + 1)

    # The 'y' coordinates for the sampling points (Contour + Normal * Distance)
    all_y = y[:, np.newaxis] + ny_unit[:, np.newaxis] * d[np.newaxis, :]
    # The 'x' coordinates for the sampling points
    all_x = x[:, np.newaxis] + nx_unit[:, np.newaxis] * d[np.newaxis, :]

    # Flatten and prepare coordinates for map_coordinates
    coords_y = all_y.T.flatten()
    coords_x = all_x.T.flatten()

    # 4. Sample the image to get the Unwrapped Image
    
    unwrapped_values = map_coordinates(image, [coords_y, coords_x], order=1, mode='nearest')

    # Reshape the result into the 2D unwrapped image strip
    unwrapped_image = unwrapped_values.reshape(2 * width + 1, num_points)

    return values_norm, unwrapped_image, smoothed_ring
