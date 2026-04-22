#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import Polcam, Unwrap, BinAveraging, UnwrappedThresholding, ThreshThick, \
    ProfileSpecBin, FitGaussian, reader, Fit_s0, LocalSegment

polariser_string = '[-45 0; 90 45]'

images = reader('..\CellsInterlinked\RubyCellSet1', 'tif')

#%% Make basic images and profiles

images_to_do = [images[0], images[4], images[16]] #usual demo images
image7 = images[7]
image4 = images[4]
image16 = images[16]
telophase_images = images[15:18]

#%%

colors, aolp, dolp, s0 = Polcam(images[3], polariser_string = polariser_string)
mask = LocalSegment(s0, erosions = 13, dilations = 0)
unwrapped = Unwrap(image = s0, mask = mask, thickness = 50, smoothing = 5, normalising = True)
profiles = BinAveraging(unwrapped, 100)

thresh_unwrapped = ThreshThick(unwrapped, block = 31, it = 2, offset = 0.005)
multiplied_image = unwrapped * thresh_unwrapped
profilesmult = BinAveraging(multiplied_image, 100)

x = np.arange(len(profiles[0]))
fits_old = []
fits_new = []
fwhms_old = []
fwhms_new = []

for i in range(len(profilesmult)):
    y = profilesmult[i]
    fitting, fwhm = FitGaussian(xdata = x, ydata = y, guess = [np.max(y), np.argmax(y), 20], FWHM = True)
    fits_old.append(fitting)
    fwhms_old.append(fwhm)
    
    y_profile = profiles[i]
    fit, FWHM = Fit_s0(x, y_profile)
    fits_new.append(fit)
    fwhms_new.append(FWHM)

#%%


# --- Plotting loop ---
plt.figure(figsize=(10, 6))

for i in range(len(profiles)):

    plt.figure(i)
    plt.plot(x, profiles[i], ls = 'solid', label = 'Original')
    plt.plot(x, fits_new[i], lw=2, ls = 'dashdot', label = 'NewFit', color = 'red')
    plt.plot(x, fits_old[i], ls = 'dashed', lw = 2, label = 'Old Fit')

    plt.legend()

#%%
print(np.round(fwhms_old, 1))
print(np.round(fwhms_new, 1))








