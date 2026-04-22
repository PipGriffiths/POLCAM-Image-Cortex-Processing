#%% Imports and setup

import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import Polcam, Unwrap, BinAveraging, LocalSegment, \
    FitGaussian, find_peaks, reader, cell_pca, cell_fft, dilationmask, ThreshThick

polariser_string = '[-45 0; 90 45]'
images = reader('..\CellsInterlinked\\NavinImages', 'tif')
chosen_indexes = [4,6,8,10,17,22,24,26,35,37,41,42,45,47,49,51,52,53,56,57,61,65,73,75,78,80,86,88,94,100,112,119,122]
savedimages = [images[i] for i in chosen_indexes]


ana = savedimages[-5]
meta = savedimages[-12]

nice = [meta, ana]

#%%

radius = 15
t = 100
b = 300

s0s = []
masks = []
unwrappeds0s = []
unwrappedffts = []
profiless0s = []
profilesffts = []
profilesmultiplieds = []

for i in range(len(nice)):        
    colors, aolp, dolp, s0 = Polcam(nice[i], polariser_string = polariser_string)
    
    s0s.append(s0)
    
    mask = LocalSegment(image = s0, erosions = 18, dilations = 0)   
    masks.append(mask)
    
    unwrappeds0 = Unwrap(s0, mask, thickness = t, smoothing = 5, normalising = False)
    unwrappeds0s.append(unwrappeds0)
    
    profiles0 = BinAveraging(unwrappeds0, bin_size = b)
    profiless0s.append(profiles0)
    
    threshed_unwrapped = ThreshThick(image = unwrappeds0, block = 31, it = 2, offset = 5)
    multiplied_image = threshed_unwrapped * unwrappeds0
    profilesmultiplied = BinAveraging(multiplied_image, bin_size = b)
    profilesmultiplieds.append(profilesmultiplied)    
    
    ffted = cell_fft(s0, radius)
    
    unwrappedfft = Unwrap(ffted, mask, thickness = t, smoothing = 5, normalising = False)
    unwrappedffts.append(unwrappedfft)
    
    profilefft = BinAveraging(unwrappedfft, bin_size = b)
    profilesffts.append(profilefft)


    
#%%

for i in range(len(s0s)):
    index = i
    plt.figure(i)
    for j in range(len(profiless0s[i])):    
        plt.plot(profilesffts[i][j])
        plt.plot(profiless0s[i][j])
        # plt.plot(profilesmultiplieds[i][j])
    # plt.imshow(s0s[i], cmap = 'gray')
    plt.title(f'Image {chosen_indexes[i]}')



