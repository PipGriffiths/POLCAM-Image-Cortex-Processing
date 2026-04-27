#Class system for my functions

import time
import numpy as np
import matplotlib.pyplot as plt
import napari
from typing import Optional, List, Union, Tuple
from matplotlib.figure import Figure

from pathlib import Path
import skimage.io
import io
from zipfile import ZipFile

# Image Processing & Signal
import skimage as ski
from skimage import measure, morphology, segmentation, filters, draw, graph
from scipy import ndimage, optimize, spatial, fft, interpolate

# Specific Napari/Qt 
from napari_polcam._functions import PolarisationCameraImage
from matplotlib.colors import hsv_to_rgb
from qtpy.QtWidgets import QApplication


class Pip_PolcamMask:
    
    def __init__(self, 
                 image: np.ndarray, 
                 polariser_string: str = '[-45 0; 90 45]', 
                 offset: int = 0) -> None:

        
        self.raw_image = image
        self.polariser_string = polariser_string
        self.offset = offset
        
        if self.offset != 0:
            self.image = self._subtract_bkgnd(self.raw_image, self.offset)
        else: self.image = self.raw_image
        
        self.s0 = None
        self.mask = None
        self.contour = None
        self.aolp = None
        self.dolp = None
        self.color = None
        
        self.long_axis_len = None
        self.short_axis_len = None
        self.long_axis_ends = None
        self.short_axis_ends = None
        self.best_block = None
        
        self.results = {}
    
    def _subtract_bkgnd(self, 
                        image: np.ndarray, 
                        offset: int) -> np.ndarray:

        '''
        Parameters
        ----------
        image : numpy array image.
        offset : background subtraction value.

        Returns
        -------
        final : numpy array image.
        '''
        image = image.astype(np.float32)
        bkgnd_corrected_img = image - offset
        final = np.clip(bkgnd_corrected_img, 0, 65535).astype(np.uint16)
        return final
    
    def Method(self, 
               method: str = 'polcam') -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        method : TYPE = str. 'polcam' or 'super-res'
            DESCRIPTION. The default is 'polcam'.

        Returns
        -------
        chooses whether to do polcam or super-res methods
        '''
        if method == 'polcam':
            self._Polcam()
        if method == 'super-res':
            self._cropper()
            self._set_top()
        else:
            raise ValueError('Please select a method. Methods: "polcam", "super-res"')
        
        return self
    
    def _cropper(self, 
                 thresh: Union[float, int] = 4, 
                 pad: Union[float, int] = 100, 
                 normalising: bool = False) -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        thresh : TYPE float/int, optional
            DESCRIPTION. The default is 4.
        pad : TYPE float/int, optional
            DESCRIPTION. The default is 100.
        normalising : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        cropped image to make super-res images smaller and quicker.
        '''
        image = self.image
        mask = image > thresh
        if not np.any(mask):
            print("No pixels found above threshold; skipping crop.")
            return self
            
        rows, cols = image.shape
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        self.image = image[max(0, y0 - pad) : min(rows, y1 + pad), 
                           max(0, x0 - pad) : min(cols, x1 + pad)]
        
        if normalising:
            img_min = self.image.min()
            img_max = self.image.max()
            if img_max > img_min:
                self.image = (self.image - img_min) / (img_max - img_min)
                
        return self

    def _set_top(self, 
                 top: Union[float, int], 
                 zeroes: bool = False) -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        top : TYPE int/float
            DESCRIPTION.
        zeroes : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        normalises image and ceilings the top x% pixels to 1
            for easier thresholding
        '''
        img_min = self.image.min()
        img_max = self.image.max()
        
        if img_max > img_min:
            self.image = (self.image - img_min) / (img_max - img_min)
            
        if zeroes:
            self.image[self.image > 0] = 1.0
        else:
            threshold = np.percentile(self.image, top)
            self.image[self.image >= threshold] = 1.0
        
        return self
    
    
    def Polcam(self) -> 'Pip_PolcamMask':
        '''
        Takes an image and performs the polcam processing.
            See https://github.com/ezrabru/napari-polcam 
            for more information.

        Returns
        -------
        4 images. All numpy arrays. colors, aolp, dolp, s0
        '''
        polcam_processor = PolarisationCameraImage(
            img = self.image, 
            method = 'Cubic spline interpolation', 
            polariser_unit = self.polariser_string, 
            offset = self.offset)
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

        hsv = np.stack([hue, sat, value], axis = -1)
        rgb = hsv_to_rgb(hsv)
        rgb = rgb*255
        HSVmap = rgb.astype(np.uint8)
        color_images = HSVmap
        
        self.color = color_images
        self.aolp = AoLP
        self.dolp = DoLP
        self.s0 = S0
        
        return self
    
    def Masking(self, 
                method: str = 'local', 
                erosions: int = 13,
                dilations: int = 0,
                radius: int = 6,
                sigma: int = 10,
                top: float = 6.0,
                properties: bool = False) -> 'Pip_PolcamMask':

        '''
        Parameters
        ----------
        method : TYPE str, optional
            Selects what kind of thresholding you want. The default is 'local'.
            If you chose super-res for the method, choose 'super'
            Allowed strings are: 'local', 'semi-auto', 'percentile', 'otsu', 'super', 'adapt local' 
        erosions : TYPE int, optional
            DESCRIPTION. The default is 13.
        dilations : TYPE int, optional
            DESCRIPTION. The default is 0.
        top : TYPE float/int, optional
            DESCRIPTION this is the top x% of pixels. The default is 6.
        radius : TYPE float/int, optional
            DESCRIPTION used in the 'otsu' method for the fft to improve accuracy. The default is 5.
        sigma : TYPE float/int, optional
            DESCRIPTION used in super-res thresholding. This is a blurring. The default is 10.
        padding : TYPE boolean, optional
            DESCRIPTION in case the cell is too close to the edge. The default is False.
        properties : TYPE boolean, optional
            DESCRIPTION gives the major and minor axis values and their coordinates. The default is False.

        Returns
        -------
        Segmented mask. And if you choose, the major minor axis properties.

        '''
        
        if self.s0 is None:
            self.Polcam()
        
        if method == 'local':
            self._LocalSegment(erosions = erosions, dilations = dilations)
        elif method == 'semi-auto':
            self._Segment_SemiAuto_Napari()
        elif method == 'percentile':
            self._Pecentile(top = top, erosions = erosions, dilations = dilations)
        elif method == 'otsu':
            self._Otsu(radius = radius)
        elif method == 'super':
            self._SuperThresholding(sigma, dilations)
        elif method == 'adapt-local':
            self._Adaptive_Local_Segment(erosion = erosions, dilation = dilations)
            
        if properties:
            self._CellProperties()
            self.results = {'long_axis_len': self.long_axis_len,
                            'short_axis_len': self.short_axis_len,
                            'long_axis_ends': self.long_axis_ends,
                            'short_axis_ends':self.short_axis_ends
                            }
        return self

        
    def _CellProperties(self) -> 'Pip_PolcamMask':
        '''
        Takes the boolean mask and finds the major and minor axes, and their end coordinates
        '''
        mask = self.mask
        
        # 1. Load and clean
        mask = mask > 0.5

        coords = np.column_stack(np.where(mask))
        
        # 2. Long Axis (Feret Diameter)
        hull = spatial.ConvexHull(coords)
        hull_points = coords[hull.vertices]
        dists = spatial.distance_matrix(hull_points, hull_points)
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
        long_axis_len = np.linalg.norm(p1-p2)
        
        longpoints = [p1[1], p2[1]], [p1[0], p2[0]]
        shortpoints = [s1[1], s2[1]], [s1[0], s2[0]]
        midpoint = midpoint[1], midpoint[0]
        
        self.long_axis_ends = longpoints
        self.long_axis_len = long_axis_len
        self.short_axis_ends = shortpoints
        self.short_axis_len = short_axis_len
        
        return self
        
    def _follow_path(self, 
                     mask: np.ndarray, 
                     p1: Tuple[int, int], 
                     p2: Tuple[int, int]) -> np.ndarray:

        '''
        Parameters
        ----------
        image : numpy array.
        p1,p2 : the points the user clicks on.
        
        Returns
        -------
        The edge of the mask as a numpy array
        '''
        image = self.s0
        grad = filters.sobel(image)
        cost = 1 / (grad + 1e-6)
        p1 = tuple(map(int, p1))
        p2 = tuple(map(int, p2))
        path, _ = graph.route_through_array(cost, p1, p2, fully_connected=True)
        return np.array(path)
    
    def _build_mask(self, image: np.ndarray, points: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Parameters
        ----------
        image : numpy array of s0.
        points : list from _follow_path.

        Returns
        -------
        mask : boolean numpy array mask. The is the segmented cell
        contour : the edge of the mask.
        '''
        paths = []
        points = np.array(points).astype(int)
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            segment = self._follow_path(image, p1, p2)
            paths.append(segment[:-1]) 
        contour = np.vstack(paths)
        mask = draw.polygon2mask(image.shape, contour)
        mask = ndimage.binary_fill_holes(mask)
        return mask, contour
    
    
    def _Segment_SemiAuto_Napari(self) -> 'Pip_PolcamMask':
        """
        Opens a napari viewer, the user clicks some points along the edge of the cell,
            6-8 is usually enough. Click Enter. 
            The _follow_path and _build_mask functions segment the cell.
            
        The manual event loop is designed to work for Jupyter and Spyder
        
        Returns: the mask and contour of the segmented cell
        """
        
        # 1. Local result container
        
        image = self.s0
        
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
                mask, contour = self._build_mask(layer_data, points_data)
                
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
        
        self.mask = results['mask']
        self.contour = contour
        
        # 5. Return captured results
        return self
            
    def _Percentile(self, 
                    top: Union[float, int], 
                    erosions: int, 
                    dilations: int) -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        top : TYPE float/int. The top percentage of pixels to keep e.g. 5
        erosions TYPE int
        dilations : TYPE int.

        Returns
        -------
        TYPE the boolean numpy array mask.
        '''
        
        image = self.s0
        
        percentile_thresh = np.percentile(image, 100 - top)  
        binary = image > percentile_thresh
        
        segment_edge = segmentation.find_boundaries(binary, mode = 'outer')
        labels = measure.label(segment_edge)
        
        cleaned = morphology.remove_small_objects(labels, min_size = 500)
        largest_edge = (cleaned == np.argmax(np.bincount(cleaned.flat)[1:]) + 1)
        
        selem = morphology.disk(1) 
        
        dilated_mask = morphology.dilation(largest_edge, selem)
        
        for j in range(dilations): #up to interpretation
            dilated_mask = morphology.dilation(dilated_mask, selem)

        filled_mask = ndimage.binary_fill_holes(dilated_mask)
        
            
        for k in range(erosions): #up to interpretation
            filled_mask = morphology.erosion(filled_mask, selem)
                
        label_image = measure.label(filled_mask)
            
        self.mask = label_image
        return self

        
    def _LocalSegment(self, 
                      erosions: int, 
                      dilations: int, 
                      padding: bool = False) -> 'Pip_PolcamMask':
        '''
        Uses local thresholding and then image cleaning to produce the mask
        
        Parameters
        ----------
        erosions : TYPE int
            DESCRIPTION.
        dilations : TYPE int
            DESCRIPTION.
        padding : TYPE boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE boolean numpy array of the mask
            DESCRIPTION.

        '''
        image = self.s0
        
        if padding:
            pad_amount = 100
            image = np.pad(image, pad_width = pad_amount, mode = 'constant', constant_values = np.min(image))
            
        block_size = 251

        selem = morphology.disk(1) 
        local_thresh = ski.filters.threshold_local(image, block_size, offset = 0.005*np.max(image))
        binary_local = image > local_thresh
        binary_local = segmentation.clear_border(binary_local)
        
        cleared_mask = morphology.remove_small_objects(binary_local, min_size = image.size*0.002)
        sealed_mask = morphology.binary_closing(cleared_mask, footprint = morphology.disk(6))
        filled_mask = ndimage.binary_fill_holes(sealed_mask)       
        filled_mask = segmentation.clear_border(filled_mask)
        
        for i in range(erosions):
            filled_mask = morphology.erosion(filled_mask, selem)
            
        filled_mask = morphology.remove_small_objects(filled_mask, min_size = image.size*0.015)

        for i in range(dilations):
            filled_mask = morphology.dilation(filled_mask, selem)            

        if padding:
            cropped_image = filled_mask[100:-100, 100:-100]
            self.mask = cropped_image
            return self

        self.mask = filled_mask
        
        return self 
     
   
    def _cell_fft(self, 
                  radius: Union[float, int]) -> np.ndarray:
        '''
        Parameters
        ----------
        radius TYPE float/int: how much of the fft signal to keep. e.g. 6

        Returns
        -------
        TYPE numpy array of the fft shifted mask.

        '''
        image = self.s0
        transform = fft.fft2(image)
        shift = fft.fftshift(transform)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[:rows, :cols]
        
        mask = (x - ccol)**2 + (y - crow)**2 >= radius**2
        shift_filtered = shift * mask
        ishift = fft.ifftshift(shift_filtered)
        img_back = fft.ifft2(ishift)
        
        return np.abs(img_back)
    
    def _Otsu(self, 
              radius: Union[float, int]) -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        radius : the radius for _cell_fft.

        Returns
        -------
        TYPE boolean numpy array mask.
        '''
        image = self.s0
        fft = self._cell_fft(image, radius = radius)
        thresh = filters.threshold_otsu(fft)
        binary = image > thresh
        cleared_mask = morphology.remove_small_objects(binary, min_size = 250)
        filled_mask = ndimage.binary_fill_holes(cleared_mask)
        self.mask = filled_mask
        return self
    
    def _SuperThresholding(self, 
                           sigma: Union[float, int], 
                           dilations: int) -> 'Pip_PolcamMask':
        '''
        Parameters
        ----------
        sigma : TYPE float/int
            DESCRIPTION.blurring factor, use maybe 10 or more
        dilations : TYPE int
            DESCRIPTION dilations before filling holes to help ndimage.binary_fill_holes.

        Returns
        -------
        TYPE boolean numpy array mask.
        '''
        image = self.s0
        selem = morphology.disk(2)
        newimg = ndimage.gaussian_filter(image, sigma)
        thresh = filters.threshold_otsu(newimg)
        otsumask = newimg > thresh
        dilated_mask = morphology.dilation(otsumask, selem)
        
        for i in range(dilations):
            dilated_mask = morphology.dilation(dilated_mask, selem)
        
        filled_mask = ndimage.binary_fill_holes(dilated_mask)
        
        for i in range(dilations):
            filled_mask = morphology.erosion(filled_mask, selem)
       
        filled_mask = morphology.remove_small_objects(filled_mask, min_size = 0.01*image.size)

        self.mask = filled_mask
        return self
    
    def _Adaptive_Local_Segment(self, 
                                erosion: int, 
                                dilation: int) -> 'Pip_PolcamMask':
        '''
        This is for finding the ideal block size (+/- 5) for local thresholding
        Parameters
        ----------
        erosion : TYPE int.
        dilation : TYPE int.

        Returns
        -------
        the boolean numpy array mask and the ideal block size.

        '''
        
        image = self.s0
        normalised_image = (image - image.min()) / (image.max() - image.min())
        
        image = ski.util.img_as_ubyte(normalised_image)
        
        best_block = None
        
        for block in range(51, 275, 10): 
            local_thresh = filters.threshold_local(image, block, offset = 5)
            binary_local = image > local_thresh
            binary_local = segmentation.clear_border(binary_local)
            cleared_mask = morphology.remove_small_objects(binary_local, min_size = 250)
            filled_mask = ndimage.binary_fill_holes(cleared_mask)
            
            if np.count_nonzero(filled_mask) > 2000:
                best_block = block
                break

        local_thresh = filters.threshold_local(image, best_block, offset = 5)
        binary_local = image > local_thresh
        binary_local = segmentation.clear_border(binary_local)
        cleared_mask = morphology.remove_small_objects(binary_local, min_size = 250)
        filled_mask = ndimage.binary_fill_holes(cleared_mask)
        
        
        selem = morphology.disk(1) 
        
        for i in range(erosion): #up to interpretation
            filled_mask = morphology.erosion(filled_mask, selem)
        
        largest_edge = segmentation.find_boundaries(filled_mask, mode='inner')
        dilated_mask = morphology.dilation(largest_edge, selem)
        
        for  i in range(dilation): #up to interpretation
            dilated_mask = morphology.dilation(dilated_mask, selem)
        self.mask = filled_mask
        self.best_block = best_block
        
        return self

class Pip_Unwrap:
    
    def __init__(self, 
                 image: np.ndarray, 
                 mask: np.ndarray, 
                 thickness: int = 100, 
                 normalising: bool = False, 
                 length: Optional[int] = None) -> None:

        self.image = image
        self.mask = mask
        self.thickness = thickness
        self.normalising = normalising
        self.length = length
        
        self.unwrapped = None
        self.startx = None
        
    def Unwrap(self, 
               thickness: Optional[int] = None, 
               smoothing: float = 5, 
               normalising: bool = False) -> 'Pip_Unwrap':

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
        
        image = self.image
        mask = self.mask
        
        thick = thickness if thickness is not None else self.thickness
        norm = normalising if normalising is not None else self.normalising
        
        contours = measure.find_contours(mask, 0.5)
        
        if not contours:
            print("No contour found.")
            return self
        
        contour = max(contours, key=len)
        contour_smooth = ndimage.gaussian_filter1d(contour, sigma=smoothing, axis=0)
        
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
        
        half_thickness = thick // 2
        d_indices = np.arange(-half_thickness, thick - half_thickness)
        s_indices = np.arange(num_points)
        
        S, D_grid = np.meshgrid(s_indices, d_indices) # Shape (thickness, perimeter)
        
        sample_y = contour_final[S, 0] + ny[S] * D_grid
        sample_x = contour_final[S, 1] + nx[S] * D_grid

        coords = np.vstack((sample_y.flatten(), sample_x.flatten()))
        
        if image.ndim == 3:    
            channels = []
            for i in range(image.shape[2]):
                sampled = ndimage.map_coordinates(image[:,:,i], coords, order = 1, mode = 'constant', cval = 0)
                channels.append(sampled.reshape(thick, num_points))
            unwrapped_image = np.stack(channels, axis = -1)
        else:        
            unwrapped_flat = ndimage.map_coordinates(image, coords, order=1, mode='constant', cval=0)
            unwrapped_image = unwrapped_flat.reshape((thick, num_points))
        
        if norm:     
            low, high = unwrapped_image.min(), unwrapped_image.max()
            if high > low:
                unwrapped_image = (unwrapped_image - low) / (high - low)
        
        self.unwrapped = unwrapped_image
    
        return self
    
    def RedefineCentre(self, 
                       start: Optional[int] = None) -> Union[Tuple[np.ndarray, int], np.ndarray]:
        '''
        Use start = None ONLY for s0 unwrapping. 
            Otherwise use the start point from the s0 unwrapping which you must do first

        Parameters
        ----------
        unwrap : TYPE
            DESCRIPTION.
        start : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION the shifted image. 
                In the case where the user inputs not s0 images then the start coordinate is the s0 one to be consistent

        '''
        if self.unwrapped is None:
            raise ValueError('Run .Unwrap() before RedefineCentre')
            
        unwrap = self.unwrapped
        
        if start is None:
            if unwrap.ndim == 3:
                intensity_profile = np.max(np.mean(unwrap, axis = 2), axis = 0)
            else:
                intensity_profile = np.max(unwrap, axis = 0)
            
            self.startx = np.argmin(intensity_profile)
        else:
            self.startx = start

        redefined = np.roll(unwrap, -self.startx, axis = 1)
        self.unwrapped = redefined

        if start is None:    
            return self.unwrapped, self.startx
        else:
            return self.unwrapped
    
    def Method(self, 
               method: str = 'interpolation', 
               target: Optional[int] = None) -> 'Pip_Unwrap':
        '''
        Parameters
        ----------
        method : TYPE str, optional
            DESCRIPTION. The default is 'interpolation'. Can also choose 'padding' or nothing
        target : TYPE int, optional
            DESCRIPTION. The default is None. Choose a target length to pad out to if None it just skips

        Returns
        -------
        TYPE numpy array
            DESCRIPTION. the padded or interpolated, (or nothing changed) image

        '''
            # Determine the target length: use passed argument, then self.length
        target_len = target if target is not None else self.length
        
        if target_len is None:
            return self

        if method == 'interpolation':
            return self._NormalisingWidth(target_len)
        elif method == 'padding':
            return self._Normaliser(target_len)
        else:
            return self
         
    def _NormalisingWidth(self, 
                          target_length: int) -> 'Pip_Unwrap':
        '''
        Parameters
        ----------
        target_length : TYPE int.
        
        Returns
        -------
        interpolated numpy array
        '''
        data = self.image
        current_length = data.shape[-1]
        old_x = np.linspace(0, 1, current_length)
        new_x = np.linspace(0, 1, target_length)
        f = interpolate.interp1d(old_x, data, kind='cubic', axis=-1)
        self.image = f(new_x)
        return self 
     
    def _Normaliser(self, 
                    target_width: int) -> 'Pip_Unwrap':
        '''
        Parameters
        ----------
        target_width : TYPE int

        Returns
        -------
        TYPE padded (with NaNs) numpy array.

        '''
        data = self.image
        current_width = data.shape[-1] # Works for both 1D and 2D
        
        # If it's already the right size (or bigger), return as is
        if current_width >= target_width:
            return self
        
        # Calculate how much padding is needed
        pad_size = target_width - current_width
        
        # Create the 'padding' block of NaNs
        # This matches the 'height' of your data automatically
        padding_shape = list(data.shape)
        padding_shape[-1] = pad_size
        nan_padding = np.full(padding_shape, np.nan)
        
        newimg = np.concatenate([data, nan_padding], axis=-1)
        self.image = newimg
        
        return self   

class Pip_Profiles:
    
    def __init__(self, 
                 image: Optional[np.ndarray] = None, 
                 binsize: int = 100, 
                 block: int = 31, 
                 it: int = 2, 
                 power: float = 2) -> None:

        
        self.image = image
        self.binsize = binsize
        self.block = block
        self.it = it
        self.power = power
        self.mask = None
        self.profiles = []
        self.split_data = []
    
    def Filter(self, 
               method: Optional[str] = 'window', 
               block: Optional[int] = None, 
               it: Optional[int] = None) -> 'Pip_Profiles':
        '''
        Choose a method to filter the profiles with.

        Parameters
        ----------
        method : TYPE str, optional. Allowed strings are 'window', 'threshold', None
            DESCRIPTION. The default is 'window'.
        block : TYPE odd int, optional
            DESCRIPTION. The default is None. block_size for the 'threshold' method
        it : TYPE int, optional
            DESCRIPTION. The default is None. Eroding and cleaning the mask iterations for 'threshold' method

        Returns
        -------
        TYPE list of 1D profiles
        '''
        b = block if block is not None else self.block
        i = it if it is not None else self.it
        
        if method == 'window':
            window_image = self._PowerWindow(self.image, power = self.power)
            self.image = window_image
            self._BinAveraging()
        elif method == 'threshold':
            self._UnwrappedThresholding(block = b, it = i)
            if self.image.ndim == 3 and self.mask.ndim == 2:
                self.image = self.image * self.mask[:,:,np.newaxis]
            else:
                self.image = self.image * self.mask
            self._BinAveraging()
        elif method == None:
            self._BinAveraging()
        
        return self
    
    def _BinAveraging(self) -> List[np.ndarray]:
        '''
        Goes through bin by bin and takes the average along each row. This is appended to a list
        '''
        if self.image is None:
            raise ValueError('No image found')
        
        bin_size = self.binsize
        image = self.image
        height, width = image.shape
        num_bins = width // bin_size
        
        self.profiles = []
        
        for i in range(num_bins):
            start_x = i * bin_size
            end_x = start_x + bin_size
            
            if end_x <= width:
                current_bin = image[:, start_x : end_x]
            else:
                # Wrap around logic: Concatenate end of image with start
                pixels_from_end = width - start_x
                pixels_from_start = bin_size - pixels_from_end
                current_bin = np.concatenate((image[:, start_x:], image[:, :pixels_from_start]), axis=1)
            
            avg_profile = np.mean(current_bin, axis = 1)
            avg_profile = np.flip(avg_profile, axis = 0)
            self.profiles.append(avg_profile)
        
        return self.profiles


    def _ThreshThick(self, 
                     image: np.ndarray, 
                     block: int, 
                     it: int, 
                     offset: int = 0, 
                     is_remainder: bool = False) -> np.ndarray:
        '''
        If method == 'threshold' this is called. Local threshold the unwrapped s0 image for a mask.
        '''
        
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
        cleared_mask = morphology.remove_small_objects(binary_local, min_size=500)
        
        selem = morphology.disk(1) 
        for _ in range(it):
            cleared_mask = morphology.erosion(cleared_mask, selem)
        
        clean_mask = morphology.remove_small_objects(cleared_mask, min_size=1000)
    
        return ndimage.binary_fill_holes(clean_mask)


    def _SplitByType(self, 
                     width: int = 100) -> 'Pip_Profiles':
        '''
        Helper function for _ThreshThick which splits the image into furrows and not furrows. Harmless on metaphase cells also.
        '''
        if self.image is None: raise ValueError('No image found')
        
        cols = self.image.shape[1]
        # Handle color images by averaging channels for the split logic
        if self.image.ndim == 3:
            col_intensities = np.mean(self.image, axis=(0, 2))
        else:
            col_intensities = np.mean(self.image, axis=0)
        
        def get_indices(center, w=width):
            return np.arange(center - w, center + w) % cols

        peak1 = np.argmax(col_intensities)
        idx1 = np.sort(get_indices(peak1))
        
        masked_intensities = col_intensities.copy()
        masked_intensities[idx1] = -1 
        peak2 = np.argmax(masked_intensities)
        idx2 = np.sort(get_indices(peak2))
        
        used_indices = np.unique(np.concatenate([idx1, idx2]))
        remainder_mask = np.ones(cols, dtype=bool)
        remainder_mask[used_indices] = False
        idx3 = np.sort(np.where(remainder_mask)[0])
        
        
        self.split_data = [
            (self.image[:, idx1], idx1),
            (self.image[:, idx2], idx2), 
            (self.image[:, idx3], idx3)
        ]
        return self
        
    def _UnwrappedThresholding(self, 
                               block: int, 
                               it: int, 
                               offset: int = 0) -> np.ndarray:
        '''
        Does the whole lot of method == 'threshold'. Uses SplitByType and ThreshThick as helpers.
        Returns: binary mask of the thresholded unwrapped s0
        '''
        if not self.split_data:
            self.SplitByType()
        
        processed_list = []
        for i, (img, coords) in enumerate(self.split_data):
            # The 3rd item (index 2) is our remainder segment
            is_rem = (i == 2)
            mask = self._ThreshThick(img, block = 31, it = 2, is_remainder=is_rem)
            processed_list.append((mask, coords))
        full_reconstruction  = np.zeros(self.image.shape, dtype = np.uint8)
        sorted_results = sorted(processed_list, key = lambda x: len(x[1]), reverse = True)
        
        for mask, coords in sorted_results:
            full_reconstruction[:, coords] = mask
        
        self.mask = full_reconstruction
        return self

    def _PeakFinderWholeImage(self, 
                              image: np.ndarray) -> List[int]:
        """
        Finds the (x, y) coordinates of the peak deviation. Uses elbow method
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


    def _PowerWindow(self, 
                     image: np.ndarray, 
                     val_min: float = 0, 
                     val_max: Optional[float] = None, 
                     target_min: int = 1, 
                     target_max: Optional[int] = None, 
                     power: float = 0.75) -> np.ndarray:
        '''
        Bases the filtering on the peak intensity in every column. The larger the value the bigger the filter at that point.
        This prevents low intensity but slow decaying signals from getting a say
        
        Returns: the filtered image.
        '''
        rows, cols = image.shape
        output = np.zeros_like(image)
        
        if val_max is None:
            val_max = (np.max(image)+1000)
        if target_max is None:
            target_max = rows
        
        target_range = target_max - target_min
        val_range = val_max - val_min if (val_max - val_min) != 0 else 1
    
        peaks = self._PeakFinderWholeImage(image)
    
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
    
class Pip_Fittings:
    
    def __init__(self, profiles: List[np.ndarray]) -> None:
        self.profiles = profiles
        self.fwhms = []
        self.fwhm_errs = []
        self.fit_data = []
        
        
    
    def Method(self, method: str = 'gauss') -> 'Pip_Fittings':
        '''
        Chooses a fitting method for the raw s0 profiles from Pip_Profiles
        Parameters
        ----------
        method : TYPE str, optional. Allowed strings are 'gauss', 'lorentz', 'sig-gauss', 'sig-lorentz', 'general'
            DESCRIPTION. The default is 'gauss'.

        Returns
        -------
        TYPE lists of numpy arrays
            DESCRIPTION.

        '''
        self.fwhms = []
        self.fwhm_errs = []
        self.fit_data = []

        for profile in self.profiles:
            # Create x_data based on the length of the current profile
            x_data = np.arange(len(profile))
            
            try:
                if method == 'gauss':
                    res = self._FitGaussian(profile, FWHM=True)
                elif method == 'lorentz':
                    res = self._FitLorentzian(profile, FWHM=True)
                elif method == 'general':
                    res = self._FitGeneralisedNormal(profile, FWHM=True)
                elif method == 'sig-gauss':
                    res = self._SigGauss(x_data, profile)
                elif method == 'sig-lorentz':
                    res = self._SigLor(x_data, profile)
                else:
                    raise ValueError(f"Method {method} not recognized.")

                self.fit_data.append(res[0])
                self.fwhms.append(res[1])
                self.fwhm_errs.append(res[2])

            except Exception as e:
                print(f"Fitting failed for a profile: {e}")
                self.fit_data.append(None)
                self.fwhms.append(np.nan)
                self.fwhm_errs.append(np.nan)
        
        return self
    
    def _PeakFinder(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, float]:
        '''
        Uses elbows to find the peak of a profile 
        '''
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
    
    def _SigLor(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        '''
        Directly fits a sigmoid + lorentzian distribution to raw s0 profiles. DO NOT USE WITH FILTERED PROFILES
        Returns: fittings, fwhm, fwhm_error
        '''

        def lorentzian(x, amp, cen, gamma):
            return amp / (1 + ((x - cen) / gamma)**2)

        def left_side_step(x, L, x0, k):
            return L / (1 + np.exp(k * (x - x0)))

        def combined_model(x, l_amp, l_cen, l_gamma, s_L, s_x0, s_k):
            return lorentzian(x, l_amp, l_cen, l_gamma) + left_side_step(x, s_L, s_x0, s_k)
        
        peak_x, peak_y = self._PeakFinder(x_data, y_data)
        plateau_y = np.mean(y_data[:int(len(x_data)*0.1)])
        
        step_back = len(x_data) * 0.15
        max_step_x = peak_x - (len(x_data) * 0.08) 
        
        p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]
    
        low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
        high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
        
        weights = np.ones(len(x_data))
        window = int(len(x_data) * 0.1) 
        weights[max(0, peak_x-window) : min(len(x_data), peak_x+window)] = 10.0
        
        # bounds = ([0, x_data[0], 0.1, 0, x_data[0], 0.1], [np.inf, x_data[-1], len(x_data), np.inf, peak_x, 2.0])
        
        popt, pcov = optimize.curve_fit(combined_model, x_data, y_data, 
                            p0=p0, bounds=(low, high), sigma=1/weights)
        
        fwhm = 2 * popt[2]
        fwhmerr = 2*np.sqrt(pcov[2,2])
        
        return combined_model(x_data, *popt), fwhm, fwhmerr

    def _SigGauss(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, float, float]:
        '''
        Directly fits a sigmoid + gaussian distribution to raw s0 profiles. DO NOT USE WITH FILTERED PROFILES
        Returns: fittings, fwhm, fwhm_error
        '''
        def gaussian(x, amp, cen, sigma):
            return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

        def left_side_step(x, L, x0, k):
            return L / (1 + np.exp(k * (x - x0)))

        def combined_model(x, g_amp, g_cen, g_sigma, s_L, s_x0, s_k):
            return gaussian(x, g_amp, g_cen, g_sigma) + left_side_step(x, s_L, s_x0, s_k)
        
        peak_x, peak_y = self._PeakFinder(x_data, y_data)
        plateau_y = np.mean(y_data[:int(len(x_data)*0.1)])
        
        step_back = len(x_data) * 0.15
        # max_step_x = peak_x - (len(x_data) * 0.08) 
        
        p0 = [peak_y, peak_x, len(x_data)*0.04, plateau_y, peak_x - step_back, 0.8]
    
        # low = [peak_y * 0.75, peak_x - (len(x_data)*0.05), 0.1, plateau_y - 0.1, 0, 0.1]
        # high = [peak_y + 0.1, peak_x + (len(x_data)*0.05), len(x_data)*0.2, plateau_y + 0.1, max_step_x, 2.0]
        
        weights = np.ones(len(x_data))
        window = int(len(x_data) * 0.1) 
        weights[max(0, peak_x-window) : min(len(x_data), peak_x+window)] = 10.0
        
        popt, pcov = optimize.curve_fit(combined_model, x_data, y_data, 
                            p0=p0, sigma=1/weights)#, bounds=(low, high))
        
        fwhm = 2.355 * popt[2]
        fwhmerr = 2.355 * np.sqrt(pcov[2,2])
        
        return combined_model(x_data, *popt), fwhm, fwhmerr

    def _FitLorentzian(self, ydata: np.ndarray, FWHM: bool = False, trim: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        '''
        Fits a lorentzian distribution to the FILTERED profiles.
        Returns: fittings, fwhm, fwhm_error
        '''
        def lorentzian(x, a, x0, gamma):
            return a * (gamma**2 / ((x - x0)**2 + gamma**2))

        ydata = np.asanyarray(ydata)
        x = np.arange(len(ydata))

        max_x, max_y = self._PeakFinder(x, ydata)    

        dist_from_center = np.abs(x - max_x)
        weights = dist_from_center + 0.001

        p0 = [max_y, max_x, 5.0]
        popt, pcov = optimize.curve_fit(lorentzian, x, ydata, p0=p0, sigma = weights)
        
        if FWHM:
            return lorentzian(x, *popt), 2 * abs(popt[2]), 2 * np.sqrt(pcov[2,2])
        return lorentzian(x, *popt)

    def _FitGaussian(self, ydata: np.ndarray, FWHM: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        '''
        Fits a gaussian distribution to the FILTERED profiles.
        Returns: fittings, fwhm, fwhm_error
        '''
        def gauss(x,a,x0,sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))
        
        xdata = np.arange(len(ydata))
        max_idx, max_val = self._PeakFinder(xdata, ydata)
        target = 5 * max_val / 6
        
        left_side = ydata[:max_idx]
        right_side = ydata[max_idx:]
        
        idx_left = np.argmin(np.abs(left_side - target))
        idx_right = np.argmin(np.abs(right_side - target)) + max_idx
            
        guess = [np.max(ydata), np.argmax(ydata), idx_right - idx_left]
        
        dist_from_center = np.abs(xdata - np.argmax(ydata))
        weights = dist_from_center + 0.5
        
        popt,pcov = optimize.curve_fit(gauss, xdata, ydata, p0 = guess, sigma = weights)

        if FWHM:
            return gauss(xdata, *popt), 2.355 * abs(popt[2]), 2.355 * np.sqrt(pcov[2,2])
        return gauss(xdata, *popt)
    
    def _FitGeneralisedNormal(self, ydata: np.ndarray, FWHM: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
        '''
        Fits a generalised normal distribution to the FILTERED profiles.
        Returns: fittings, fwhm, fwhm_error
        '''
        # a: amplitude, x0: center, alpha: scale (width), beta: shape
        def gen_normal(x, a, x0, alpha, beta):
            return a * np.exp(-(np.abs(x - x0) / alpha) ** beta)
        
        xdata = np.arange(len(ydata))
        max_x, max_y = self._PeakFinder(xdata, ydata)
        
        # Initial guess
        # beta=2.5 is a good starting point for a flat-topped peak
        guess = [max_y, max_x, 5.0, 2.5] 
        
        popt, pcov = optimize.curve_fit(gen_normal, xdata, ydata, p0=guess)
        fwhm_val = 2 * popt[2] * (np.log(2))**(1/popt[3])
        if FWHM:
            # Error estimation for GenNormal FWHM is complex; returning sigma-alpha error as proxy
            return gen_normal(xdata, *popt), fwhm_val, np.sqrt(pcov[2,2])
        return gen_normal(xdata, *popt)
           
def reader(path: Union[str, Path], 
           imagetype: str) -> List[np.ndarray]:
    '''
    Reads an entire folder of images into a list from a filepath and an image type to search for.
    '''
    images = []
    files = Path(rf'{path}').rglob(rf'*.{imagetype}')
    for file in files:
        images.append(skimage.io.imread(file))
    return images

def figuresaver(imagelist: List[Union[np.ndarray, Figure]], 
                foldername: str, 
                typename: str, 
                formatting: str = 'png', 
                original: bool = False) -> None:
    '''
    Saves images or figures as a zipped folder using a numerical naming system.
    '''
    
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

        
        
        
        
        
        