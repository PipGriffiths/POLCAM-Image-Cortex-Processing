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
