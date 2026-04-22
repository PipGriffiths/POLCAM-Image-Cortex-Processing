# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:23:54 2026

@author: griff
"""

import numpy as np
#%%
from napari_polcam._functions import PolarisationCameraImage
#%%
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