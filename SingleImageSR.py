#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import Unwrap, BinAveraging, SuperThresholding, \
    ProfileSpecBin, FitGaussian, find_peaks, reader, LocalSegment, set_top, cropper


superimages = reader('..\CellsInterlinked\SuperClear', 'tif')
#vmax=np.percentile(cropped, 95) use this argument whenever doing plt.imshow


#%% Rubys Images
rubys = superimages[5]

image = rubys
cropped = cropper(image, normalising = True)
newcrop = set_top(cropped, top = 97.5, zeroes=False)
dilated, mask = SuperThresholding(newcrop, sigma = 10, dilations = 30, redilations = 50)
bestimage = dilated * cropped

rubyunwrap = Unwrap(image = image, mask = mask, thickness = 600, smoothing = 5)
rubyprofile = BinAveraging(rubyunwrap, bin_size = 100)

fittings = []
fwhms = []
for i in range(len(rubyprofile)):
    fitting, fwhm = FitGaussian(ydata = rubyprofile[i], FWHM = True)
    fittings.append(fitting)
    fwhms.append(fwhm)

#%% nice plot of all but profiles
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(12, 10)) 
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) 

# 2. Add the Top-Left Plot
ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(image, cmap='gray')
ax0.set_title("Original Image", fontsize=20)
ax0.axis('off')

# 3. Add the Top-Right Plot
ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(mask, cmap='gray')
ax1.set_title("Mask", fontsize=20)
ax1.axis('off')

# 4. Add the Bottom Plot
ax2 = fig.add_subplot(gs[1, :]) 
ax2.imshow(rubyunwrap, cmap='gray')
ax2.set_title(r"Unwrapped $S_0$", fontsize=20)

plt.tight_layout()
plt.subplots_adjust(hspace=-0.35)





#%%
chosenidxs = [-4,14]


fig, ax = plt.subplots(1,1)
for j in range(len(chosenidxs)):
    ax.plot(rubyprofile[chosenidxs[j]], color = 'b', label = 'Profile')
    ax.plot(fittings[chosenidxs[j]], color = 'r', label = f'Fitting')#, FWHM = {np.round(fwhms[j], 2)} pixels')
    if j == 0:
        ax.legend(loc = 'upper right')
ax.set_title(f'Super Resolution Profiles, bin size = 600 pixels', fontsize = 14)
ax.set_ylabel('Intensity', fontsize = 13)
ax.set_xlabel('Thickness', fontsize = 13)

print(fwhms[chosenidxs[0]])
print(fwhms[chosenidxs[1]])

# print(fwhms)
# print(max(fwhms))
# print(min(fwhms))

