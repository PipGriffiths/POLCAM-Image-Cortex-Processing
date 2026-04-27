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
