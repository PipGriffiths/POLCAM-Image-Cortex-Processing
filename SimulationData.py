import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data")
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.io
import numpy as np

#Custom functions
from All_Segmenting_Functions import Polcam, subtract_bkgnd,\
    Segment_Auto_Local, Unwrap_General_Contour, BinAveraging, Unwrap_NotNorm, \
        ThreshThick, counter, HWHM, SplitByType, xticks, CombineThresholded, \
            CortexSim

synthetic = CortexSim(width = 2500, height = 100, 
                                         cortex_thickness=25, thickness_randomness=6, blur_sigma=3 )

plt.imshow(synthetic, cmap = 'gray')


#%%

import numpy as np
from scipy.ndimage import gaussian_filter

def CortexSim(width, height, cortex_thickness, thickness_randomness, blur_sigma):
    img = np.zeros((height, width))
    centerline = np.zeros(width)
    current_y = height // 2
    
    reversion_strength = 0.015 
    random_step_size = 1.5     
    
    # --- Generate Centerline ---
    for i in range(1, width):
        pull = (height // 2 - current_y) * reversion_strength
        current_y += pull + np.random.normal(0, random_step_size)
        centerline[i] = current_y
    centerline = gaussian_filter(centerline, sigma=15)
    
    # --- NEW: Gaussian Brightness Profile ---
    # Define centers at 1/4 and 3/4
    mu1, mu2 = width * 0.25, width * 0.75
    # sigma_bright controls how "spread out" the bright spots are
    sigma_bright = width * 0.12 
    
    x = np.arange(width)
    # Calculate two Gaussians
    g1 = np.exp(-(x - mu1)**2 / (2 * sigma_bright**2))
    g2 = np.exp(-(x - mu2)**2 / (2 * sigma_bright**2))
    
    # Combine them (taking the max or sum) and normalize
    brightness_profile = np.maximum(g1, g2) 
    # Optional: ensure a baseline brightness so the line doesn't disappear entirely
    brightness_profile = 0.2 + 0.8 * brightness_profile 

    # --- Render Image ---
    for i in range(width):
        # Using a Gaussian for thickness as requested in your comment
        local_thick = cortex_thickness + np.random.normal(0, thickness_randomness)
        
        y_indices = np.arange(height)
        dist = np.abs(y_indices - centerline[i])
        
        # Cross-sectional intensity
        column_intensity = np.exp(-(dist**2) / (2 * (max(0.1, local_thick)/3)**2))
        
        img[:, i] = column_intensity * brightness_profile[i]

    img = gaussian_filter(img, sigma=blur_sigma)
    noise = np.random.normal(0, 0.05, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return (img * 255).astype(np.uint8)

