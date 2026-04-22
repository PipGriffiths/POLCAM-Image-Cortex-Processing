#%%
#Spare single image results --- script is used in conjuction with SingleImageResults.py

from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import numpy as np
from Useless_Functions import Segment_Auto_Global, Adaptive_Local_Segment, HWHM

from Tidy_Functions import reader, Polcam, LocalSegment, Unwrap, RedefineCentre, \
    NormalisingWidth, BinAveraging, ThreshThick, FitGaussian, \
        PeakFinderWholeImage, PeakFinder, PowerWindow, Window, FitLorentzian, \
            SigGauss, SigLor

rubys = reader('..\CellsInterlinked\RubyCellSet1', 'tif')
chosen = rubys[8]

size = 100
thick = 100
    
polariser_string = '[-45 0; 90 45]'
color, aolp, dolp, s0 = Polcam(chosen, polariser_string = polariser_string)
lmask = LocalSegment(image = s0, erosions = 13, dilations = 0)   
unwrappeds0 = Unwrap(s0, lmask, thickness = thick, smoothing = 5, normalising = False)
unwrappeds0, startx = RedefineCentre(unwrappeds0)  
maxcoords = PeakFinderWholeImage(unwrappeds0)
longprofile = unwrappeds0[maxcoords, np.arange(len(maxcoords))]


# basic version

basic_profiles = BinAveraging(unwrappeds0, bin_size = size)
basic_threshold = ThreshThick(image = unwrappeds0, block = 31, it = 2, offset = 5)
basic_multipliedimage = basic_threshold * unwrappeds0
multiplied_profiles = BinAveraging(image = basic_multipliedimage, bin_size = size)

basic_gauss_fitting = []
basic_gauss_fwhm = []
basic_lorentz_fitting = []
basic_lorentz_fwhm = []

for j in range(len(multiplied_profiles)):
    y = multiplied_profiles[j]
    gaussfit, gaussfwhm = FitGaussian(ydata = y, FWHM = True)
    lorentzfit, lorentzfwhm = FitLorentzian(ydata = y, FWHM = True)
    basic_gauss_fitting.append(gaussfit)
    basic_gauss_fwhm.append(gaussfwhm)
    basic_lorentz_fitting.append(lorentzfit)
    basic_lorentz_fwhm.append(lorentzfwhm)  

# new dynamic ROI method with basic window and thresholding the window also

ROI_window = PowerWindow(unwrappeds0, val_min = 0, val_max = 2*np.max(unwrappeds0), 
                         target_min = 1, target_max = len(unwrappeds0), power = 2)
ROI_profiles = BinAveraging(ROI_window, bin_size = size)

ROI_gauss_fitting = []
ROI_gauss_fwhm = []
ROI_lorentz_fitting = []
ROI_lorentz_fwhm = []

for j in range(len(ROI_profiles)):
    y = ROI_profiles[j]
    gaussfit, gaussfwhm = FitGaussian(ydata = y, FWHM = True)
    lorentzfit, lorentzfwhm = FitLorentzian(ydata = y, FWHM = True)
    ROI_gauss_fitting.append(gaussfit)
    ROI_gauss_fwhm.append(gaussfwhm)
    ROI_lorentz_fitting.append(lorentzfit)
    ROI_lorentz_fwhm.append(lorentzfwhm)


numbers = [1,3,5,7,9,11,13]
demo_profs = [basic_profiles[i] for i in numbers]
demo_mults = [multiplied_profiles[i] for i in numbers]

newnumbers = [7,11]
newprofs = [basic_profiles[i] for i in newnumbers]
newmults = [multiplied_profiles[i] for i in newnumbers]
newgaus = [basic_gauss_fitting[i] for i in newnumbers]
newlor = [basic_lorentz_fitting[i] for i in newnumbers]

gausses, siggs, sg_fwhms = SigGauss(np.arange(100), basic_profiles)
lorentzes, sigls, sl_fwhms = SigLor(np.arange(100), basic_profiles)

demo_sg = [[gausses[i], siggs[i]] for i in newnumbers]
demo_sl = [[lorentzes[i], sigls[i]] for i in newnumbers]
demo_sg_fwhms = [sg_fwhms[i] for i in newnumbers]
demo_sl_fwhms = [sl_fwhms[i] for i in newnumbers]


#%%
from Tidy_Functions import Otsu, Segment_SemiAuto_Napari

omask = Otsu(s0)
pmask = Segment_Auto_Global(s0, top = 5.4)
smask, scontour = Segment_SemiAuto_Napari(s0)

demoimages = [rubys[13], rubys[8], rubys[16]]
#%% Demo Cell Phases
fig, ax = plt.subplots(1,3)
ax[0].imshow(demoimages[1], cmap = 'gray')
ax[0].set_title('Metaphase')
ax[0].axis('off')

ax[1].imshow(demoimages[0], cmap = 'gray')
ax[1].set_title('Anaphase')
ax[1].axis('off')

ax[2].imshow(demoimages[2], cmap = 'gray')
ax[2].set_title('Telophase')
ax[2].axis('off')

labels = ['a)', 'b)', 'c)']
for i, j in enumerate(ax):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='bottom', 
            ha='left')


#%% POLCAM Results

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(s0, cmap = 'gray')
ax[0,0].set_title(r"$S_0$")
ax[0,0].axis('off')

ax[0,1].imshow(color)
ax[0,1].set_title('RGB Image')
ax[0,1].axis('off')

ax[1,0].imshow(dolp, cmap = 'gray')
ax[1,0].set_title('DoLP')
ax[1,0].axis('off')

ax[1,1].imshow(aolp, cmap = 'gray')
ax[1,1].set_title('AoLP')
ax[1,1].axis('off')


labels = ['a)', 'b)', 'c)', 'd)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='bottom', 
            ha='left')

fig.subplots_adjust(wspace=-0.5, hspace=0.25)

#%% Masks Separate

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(smask, cmap = 'gray')
ax[0,0].set_title("Semi-Automatic Napari", fontsize = 11)
ax[0,0].axis('off')

ax[0,1].imshow(lmask, cmap = 'gray')
ax[0,1].set_title('Local Thresholding', fontsize = 11)
ax[0,1].axis('off')

ax[1,1].imshow(omask, cmap = 'gray')
ax[1,1].set_title('Otsu Thresholding', fontsize = 11)
ax[1,1].axis('off')

ax[1,0].imshow(pmask, cmap = 'gray')
ax[1,0].set_title('Percentile Thresholding', fontsize = 11)
ax[1,0].axis('off')

labels = ['a)', 'b)', 'c)', 'd)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='bottom', 
            ha='left')

fig.subplots_adjust(wspace=-0.35, hspace=0.25)


#%% Masks Overlaid
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


masked_lmask = np.ma.masked_where(lmask == 0, lmask)
masked_pmask = np.ma.masked_where(pmask == 0, pmask)
masked_smask = np.ma.masked_where(smask == 0, smask)
masked_omask = np.ma.masked_where(omask == 0, omask)

cmap00 = ListedColormap(['white'])
cmap01 = ListedColormap(['green'])
cmap10 = ListedColormap(['red'])
cmap11 = ListedColormap(['blue'])

fig, ax = plt.subplots(1, 2, facecolor = 'white')
ax[0].set_facecolor('black')
ax[0].set_title('Mask Overlays')
ax[1].set_facecolor('black')
ax[1].set_title(r'Semi-Auto overlaid $S_0$')

# Display the images (removed the 'label' argument as it's not supported here)
# cmap00 with Semi-Auto
# cmap01 with Local
# cmap10 with percentile
# cmap11 with otsu

ax[0].imshow(masked_omask, cmap=cmap11)
ax[0].imshow(masked_pmask, cmap=cmap10)
ax[0].imshow(masked_lmask, cmap=cmap01)
ax[0].imshow(masked_smask, cmap=cmap00)

# Create manual legend handles
# We use the second color of each colormap to represent the "active" mask color
legend_handles = [
    mpatches.Patch(color='white', label='Semi-Auto'),
    mpatches.Patch(color='green', label='Local'),
    mpatches.Patch(color='blue', label='Otsu'),
    mpatches.Patch(color='red', label='Percentile')
]

# Apply legend and turn off axis
ax[0].legend(handles=legend_handles, loc='lower left', fontsize=8)
# ax[0].axis('off')

ax[1].imshow(smask, cmap = 'gray')
ax[1].imshow(s0, cmap = 'gray', alpha = 0.65)


handle = [
    mpatches.Patch(color = 'White', label = 'Semi-Auto'), 
    mpatches.Patch(color = 'gray', label = r'$S_0$') ]

# ax[1].axis('off')
ax[1].legend(handles = handle, loc = 'lower left', fontsize = 8)

labels = ['a)', 'b)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.95, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='bottom', 
            ha='right')

for a in ax.flatten():
    # 1. Hide the tick marks and labels, but keep the axis 'on'
    a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # 2. Iterate through the four spines (top, bottom, left, right)
    for spine in a.spines.values():
        spine.set_visible(True)     # Ensure they are shown
        spine.set_color('white')    # Make them white
        spine.set_linewidth(2)

fig.subplots_adjust(wspace=0.05, hspace=0.25)

#%% Raw s0 and unwrapped

cb_palette = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7"   # Purple
]

fig, ax = plt.subplots(2,1)

for i in range(len(demo_profs)):   
    ax[0].plot(demo_profs[i], label = f'Bin {i+1}', color = cb_palette[i])
ax[0].legend(fontsize='small')
ax[0].set_title(r'$S_0$ Profiles')
ax[0].set_ylabel('Intensity')
ax[0].set_xlabel('Thickness')

x = np.arange(0, len(unwrappeds0[0])-(size/2), size) + (size/2)
demo_x = [int(x[(2*i)+1]) for i in range(len(x)//2)]
bin_x = [int(x[(2*i)]) for i in range(len(x)//2 + 1)]

ax[1].imshow(unwrappeds0, cmap = 'gray')
ax[1].set_title(r'Unwrapped $S_0$')
ax[1].tick_params(axis='both', labelsize=8)
ax[1].set_ylabel('Thickness')
ax[1].set_xlabel('Perimeter')

for i in range(len(demo_x)):
    ax[1].axvline(demo_x[i], color=cb_palette[i], linestyle='solid', linewidth=1)
    ax[1].axvline(bin_x[i], color='r', linestyle='--', linewidth=0.5)
    if i == 0:
        ax[1].axvline(bin_x[-1], color='r', linestyle='--', linewidth=0.5)

fig.subplots_adjust(wspace = 0.05, hspace = 0.05)

#%% Masks for profiles

fig, ax = plt.subplots(3,1, sharex = True)

ax[0].imshow(unwrappeds0, cmap = 'gray')
ax[1].imshow(basic_threshold, cmap = 'gray')
ax[2].imshow(basic_multipliedimage, cmap = 'gray')
ax[2].set_xlabel('Perimeter')
ax[1].set_ylabel('Thickness')
ax[0].set_title(r'Unwrapped $S_0$')
ax[1].set_title('Local Threshold Mask')
ax[2].set_title('Multiplied Image')

labels = ['a)', 'b)', 'c)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.99, 0.95, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='top', 
            ha='right')

fig.subplots_adjust(wspace=0.0, hspace=-0.65)

#%% Fittings from mask multiplication
fig, ax = plt.subplots(1,3, sharey = True)

colors2 = ['black', "#009E73"]

for i in range(len(newnumbers)):
    ax[0].plot(newprofs[i], label = f'Bin {newnumbers[i]}', color = colors2[i])
    ax[0].plot(newmults[i], color = "#56B4E9")

ax[1].plot(newprofs[0], color = 'black', label = f'Bin {newnumbers[0]}')
ax[1].plot(newmults[0], color = "#56B4E9", label = 'Mask Multiplied')
ax[1].plot(newgaus[0], color = 'blue', label = 'Gaussian Fit')
ax[1].plot(newlor[0], color = 'red', label = 'Lorentzian Fit')

ax[2].plot(newprofs[1], color = "#009E73", label = f'Bin {newnumbers[1]}')
ax[2].plot(newmults[1], color = "#56B4E9")
ax[2].plot(newgaus[1], color = 'blue')
ax[2].plot(newlor[1], color = 'red')   

ax[1].tick_params(labelleft=False)
ax[1].tick_params(left = False)
ax[2].tick_params(labelleft=False)
ax[2].yaxis.tick_right()
ax[2].tick_params(labelright=True) 

ax[1].spines['left'].set_visible(False)
ax[2].spines['left'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[0].spines['right'].set_visible(False)

ax[0].legend(loc = 'upper right', fontsize = 'x-small')
ax[1].legend(loc = 'upper right', fontsize = 'x-small')
ax[2].legend(loc = 'upper right', fontsize = 'x-small')

ax[0].set_ylabel('Intensity')
ax[1].set_xlabel('Thickness [pixels]')

labels = ['a)', 'b)', 'c)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.95, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='black', 
            va='top', 
            ha='left')

ax[0].set_title('Mask Multiplied')
ax[1].set_title('Pole Fitting')
ax[2].set_title('Furrow Fitting')
fig.suptitle('Thresholding and Fitting for Pole and Furrow Regions', y = 1.01, fontsize = 14)

fig.subplots_adjust(wspace = 0.015)

#%% ROI mask

plt.imshow(ROI_window, cmap = 'gray')
plt.title('Dynamic ROI Mask')
plt.xlabel('Perimeter [pixels]')

#%% fft impact
from Tidy_Functions import cell_fft
from Useless_Functions import Segment_Auto_Global

newimg = cell_fft(image = s0, radius = 4)
fftmask = Segment_Auto_Global(newimg, 3)

fig, ax = plt.subplots(1, 2, dpi = 300)
ax[1].imshow(fftmask, cmap = 'gray')
ax[0].imshow(newimg, cmap = 'gray')#, alpha = 0.5)
ax[1].imshow(s0, cmap = 'gray', alpha = 0.5)
ax[0].axis('off')
ax[1].axis('off')
ax[0].set_title('Fast Fourier Filtered')
ax[1].set_title('Stokes Mask Overlay')
fig.tight_layout()

labels = ['a)', 'b)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.95, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='white', 
            va='bottom', 
            ha='right')

#%% direct fitting
gausses, siggs, sg_fwhms = SigGauss(np.arange(100), basic_profiles)
lorentzes, sigls, sl_fwhms = SigLor(np.arange(100), basic_profiles)

demo_sg = [[gausses[i], siggs[i]] for i in newnumbers]
demo_sl = [[lorentzes[i], sigls[i]] for i in newnumbers]
demo_sg_fwhms = [sg_fwhms[i] for i in newnumbers]
demo_sl_fwhms = [sl_fwhms[i] for i in newnumbers]

color2 = ['black', 'blue', 'red']

fig, ax = plt.subplots(2,2, sharey = True)

ax[0,0].plot(newprofs[0], color = 'black', label = f'Bin {newnumbers[0]}')
ax[0,0].plot(demo_sg[0][0], color = 'blue', label = 'Gaussian Fit')
ax[0,0].plot(demo_sg[0][1], color = 'red', label = 'Sigmoid Fit')

ax[0,1].plot(newprofs[1], color = 'black', label = f'Bin {newnumbers[1]}')
ax[0,1].plot(demo_sg[1][0], color = 'blue')
ax[0,1].plot(demo_sg[1][1], color = 'red')

ax[1,0].plot(newprofs[0], color = "#009E73", label = f'Bin {newnumbers[0]}')
ax[1,0].plot(demo_sl[0][0], color = 'blue', label = 'Lorentzian Fit')
ax[1,0].plot(demo_sl[0][1], color = 'red', label = 'Sigmoid Fit')   

ax[1,1].plot(newprofs[1], color = "#009E73", label = f'Bin {newnumbers[1]}')
ax[1,1].plot(demo_sl[1][0], color = 'blue')
ax[1,1].plot(demo_sl[1][1], color = 'red')   


ax[0,0].legend(loc = 'upper right')
ax[1,0].legend(loc = 'upper right')
ax[0,1].legend(loc = 'upper right')
ax[1,1].legend(loc = 'upper right')

fig.supylabel('Intensity')
fig.supxlabel('Thickness [pixels]')

labels = ['a)', 'b)', 'c)', 'd)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.95, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='black', 
            va='top', 
            ha='left')

ax[0,0].set_title('SG Polar')
ax[0,1].set_title('SG Furrow')
ax[1,0].set_title('SL Pole')
ax[1,1].set_title('SL Furrow')

fig.suptitle('Direct Fitting', y = 1.01, fontsize = 14)

fig.subplots_adjust(wspace = 0.15, hspace = 0.35)


#%% direct fitting perimeter

def forward(x):
    return x*57.5
def inverse(x):
    return x / 57.5

x = np.arange(0, len(unwrappeds0[0])-(size/2), size) + (size/2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot your data
ax1.plot(x, basic_gauss_fwhm, label = 'Threshold Gauss', color = 'blue')
ax1.scatter(x, basic_gauss_fwhm, s = 3)

ax1.plot(x, basic_lorentz_fwhm, label = 'Threshold Lorentz', color = 'red')
ax1.scatter(x, basic_lorentz_fwhm, s = 3)

ax1.plot(x, ROI_gauss_fwhm, label = 'ROI Gauss', color = 'orange')
ax1.scatter(x, ROI_gauss_fwhm, s = 3, color = 'orange')

ax1.plot(x, ROI_lorentz_fwhm, label = 'ROI Lorentz', color = 'black')
ax1.scatter(x, ROI_lorentz_fwhm, s = 3)

ax1.plot(x, sg_fwhms, label = 'Direct SG')
ax1.scatter(x, sg_fwhms, s = 3)

ax1.plot(x, sl_fwhms, label = 'Direct SL')
ax1.scatter(x, sl_fwhms, s = 3)

ax1.set_ylabel('FWHM [pixels]')
secax = ax1.secondary_yaxis('right', functions=(forward, inverse))
secax.set_ylabel('FWHM [nm]')
ax1.legend(loc = 'upper left')
# ax1.set_ylim([0,20])

ax2.imshow(unwrappeds0, cmap = 'gray', extent=[0, len(unwrappeds0[0]), 0, len(unwrappeds0)])
fig.subplots_adjust(hspace = -0.38) 
binlines = [size*i for i in range(len(x))]
for i in range(len(binlines)):
    ax1.axvline(binlines[i], color='r', linestyle='--', linewidth=0.5)
    ax2.axvline(binlines[i], color='r', linestyle='--', linewidth=0.5)
ax2.set_xlabel('Perimeter [pixels]')

fig.suptitle('FWHM Corresponding to their Bin', y = 0.94, fontsize = 14)

#%% adaptive block sizes

import skimage as ski
import skimage.segmentation as ski_seg
import skimage.morphology as ski_morph
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries
from scipy.spatial.distance import dice
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


def AdaptiveSegment(s0, blocks):
    
    adapt_masks = []
    image = s0
    
    normalised_image = (image - image.min()) / (image.max() - image.min())
    
    image = ski.util.img_as_ubyte(normalised_image)
    for block in blocks:    
        local_thresh = ski.filters.threshold_local(image, block, offset = 5)
        binary_local = image > local_thresh
        binary_local = ski_seg.clear_border(binary_local)
        cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = 250)
        filled_mask = binary_fill_holes(cleared_mask)
        selem = ski_morph.disk(1) 
        
        for i in range(13): #up to interpretation
            filled_mask = ski_morph.erosion(filled_mask, selem)
        adapt_masks.append(filled_mask)
        print(block)
    return adapt_masks

blocks = np.arange(1, 4001, 50)
adaptive_masks = AdaptiveSegment(s0s, blocks)
#Use smask as ground truth

smask_flat = smask.flatten()
flat_masks = [adaptive_masks[i].flatten() for i in range(len(adaptive_masks))]
dices = [1 - dice(smask_flat, flat_masks[i]) for i in range(len(flat_masks))]

plt.plot(blocks[:], dices[:], lw = 1)
plt.scatter(blocks[:], dices[:], s = 2)
plt.xlabel('Block size')
plt.ylabel('Dice Coefficient')
plt.title('Dice Coefficients vs Different Block Sizes')

from scipy.spatial.distance import dice

#Use smask as ground truth
print(1 - dice(smask.flatten(), lmask.flatten()))
print(1 - dice(smask.flatten(), omask.flatten()))
print(1 - dice(smask.flatten(), pmask.flatten()))

