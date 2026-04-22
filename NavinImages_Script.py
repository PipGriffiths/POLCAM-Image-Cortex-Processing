#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import Polcam, Unwrap, BinAveraging, LocalSegment, \
    counter, UnwrappedThresholding, ThreshThick, ProfileSpecBin, FitGaussian, find_peaks, reader, figuresaver

polariser_string = '[-45 0; 90 45]'
navinimages = reader('..\CellsInterlinked\\NavinImages', 'tif')

chosen_indexes = [4,6,8,10,17,22,24,26,35,37,41,42,45,47,49,51,52,53,56,57,61,65,73,75,78,80,86,88,94,100,112,119,122]
savedimages = [navinimages[i] for i in chosen_indexes]

from GigaFunction import Everything
data, keys = Everything(savedimages, masks=True)
print(data.keys())

unwrappeds0s = data.get('unwrapped_s0')
unwrappedaolp = data.get('unwrapped_aolp')
unwrappeddolps = data.get('unwrapped_dolp')
maxcoords = data.get('max_coords')
xaxes = data.get('x_axes')
masks = data.get('masks')

meta_idx = [3, 5, 6, 7, 8, 9, 11, 17, 18, 19, 20, 21, 22, 26, 31, 32]
ana_idx = [0, 1, 2, 4, 10, 12, 13, 15, 23, 25, 28, 29]
telo_idx = [14, 16, 24, 27, 30]

metas = [unwrappeds0s[i] for i in meta_idx]
anas = [unwrappeds0s[i] for i in ana_idx]
telos = [unwrappeds0s[i] for i in telo_idx]

# figuresaver(imagelist = , foldername = '', typename = 'Cell')



