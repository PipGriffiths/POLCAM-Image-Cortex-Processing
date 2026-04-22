#%%
# Manual SingleImageResults not using GigaFunction
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import reader, Polcam, LocalSegment, Unwrap, RedefineCentre, \
    NormalisingWidth, PeakFinderWholeImage, PeakFinder, Normaliser, PowerWindow


rubys = reader('..\CellsInterlinked\RubyCellSet1', 'tif')
chosen = rubys[8]

size = 100
thick = 100
    
polariser_string = '[-45 0; 90 45]'
color, aolp, dolp, s0 = Polcam(chosen, polariser_string = polariser_string)
lmask = LocalSegment(image = s0, erosions = 13, dilations = 0)   
unwrappeds0 = Unwrap(s0, lmask, thickness = thick, smoothing = 5, normalising = False)
unwrappeds0, startx = RedefineCentre(unwrappeds0)  


unwrappedaolp = Unwrap(aolp, lmask, thick, 5)
unwrappedaolp = RedefineCentre(unwrappedaolp, startx)

unwrappeddolp = Unwrap(dolp, lmask, thick, 5)
unwrappeddolp = RedefineCentre(unwrappeddolp, startx)

maxcoords = PeakFinderWholeImage(unwrappeds0)
longprofile = unwrappeds0[maxcoords, np.arange(len(maxcoords))]


ROI_window = PowerWindow(unwrappeds0, val_min = 0, val_max = 2*np.max(unwrappeds0), 
                         target_min = 1, target_max = len(unwrappeds0), power = 2)
ROImask = (ROI_window != 0)
dolpcortex = unwrappeddolp * ROImask
dolpcortex[dolpcortex == 0] = np.nan
dolpaverages = np.nanmean(dolpcortex, axis=0)

#%%
plt.plot(dolpaverages)
plt.title('Long DoLP Profile ROI Averaged')
plt.ylabel('DoLP')
plt.xlabel('Perimeter')

longprofile = unwrappeddolp[maxcoords, np.arange(len(maxcoords))]
# plt.plot(longprofile)

#%% aolp and dolp unwrapping

fig, ax = plt.subplots(3,1)

ax[0].imshow(unwrappeds0, cmap = 'gray')
ax[0].set_title(r'Unwrapped $S_0$')

ax[1].imshow(unwrappedaolp, cmap = 'gray')
ax[1].set_title('Unwrapped AoLP')

ax[2].imshow(unwrappeddolp, cmap = 'gray')
ax[2].set_title('Unwrapped DoLP')

fig.supxlabel('Perimeter', y = 0.15)
fig.supylabel('Thickness', x = 0.05)
fig.suptitle('Unwrapped Images for one cell', y = 0.85)
fig.subplots_adjust(wspace=0.05, hspace=-0.45)

labels = ['a)', 'b)', 'c)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.01, 1.55, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='black', 
            va='top', 
            ha='left')

#%%
    
aolpimage = unwrappedaolps
dolpimage = unwrappeddolps
s0img = unwrappeds0s
x = np.arange(len(aolpimage[0]))
    
pointsaolp = []
pointss0 = []
pointsdolp = []
num = maxcoords
 
for j in range(len(num)):
    coord = num[j]
    pointaolp = aolpimage[coord]
    pointsaolp.append(pointaolp)
    pointsdolp.append(dolpimage[coord])
    points0 = s0img[coord]
    pointss0.append(points0)

    
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))

# ax1.plot(x, pointsaolp)
ax1.imshow(unwrappedaolps, cmap = 'gray')
ax1.set_title('AoLP')

# ax2.plot(x, pointsdolp)
ax2.imshow(unwrappeddolps, cmap = 'gray')
ax2.set_title('DoLP')

# ax3.plot(x, pointss0)
ax3.imshow(unwrappeds0s, cmap = 'gray')
ax3.set_title('s0')

fig.tight_layout()



