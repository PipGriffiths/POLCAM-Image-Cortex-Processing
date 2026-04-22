#%%
#Normalising methods SingleImageResults

import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import reader, Polcam, LocalSegment, Unwrap, RedefineCentre, \
    Normaliser, PeakFinderWholeImage, PeakFinder, NormalisingWidth

rubys = reader('..\CellsInterlinked\RubyCellSet1', 'tif')
chosen = rubys[8]


size = 100
thick = 100
  
demoimages = [rubys[14], rubys[8], rubys[17]]
  
polariser_string = '[-45 0; 90 45]'
aolps = []
dolps = []
s0s = []

for i in range(len(demoimages)):
    color, aolp, dolp, s0 = Polcam(demoimages[i], polariser_string = polariser_string)
    mask = LocalSegment(image = s0, erosions = 13, dilations = 0)   
    basic_unwrappeds0 = Unwrap(s0, mask, thickness = thick, smoothing = 5, normalising = False)
    basic_unwrappeds0, startx = RedefineCentre(basic_unwrappeds0)  
    
    basic_unwrappedaolp = Unwrap(aolp, mask, thick, 5)
    basic_unwrappedaolp = RedefineCentre(basic_unwrappedaolp, startx)
    
    basic_unwrappeddolp = Unwrap(dolp, mask, thick, 5)
    basic_unwrappeddolp = RedefineCentre(basic_unwrappeddolp, startx)
    
    
    s0s.append(basic_unwrappeds0)
    aolps.append(basic_unwrappedaolp)
    dolps.append(basic_unwrappeddolp)
    

longs0s = []
longaolps = []
longdolps = []
maxcoords = []

for i in range(len(s0s)):
    maxcoords = PeakFinderWholeImage(s0s[i])
    
    longs0 = s0s[i][maxcoords, np.arange(s0s[i].shape[1])]
    longs0s.append(longs0)

    longaolp = aolps[i][maxcoords, np.arange(aolps[i].shape[1])]
    longaolps.append(longaolp)
    
    longdolp = dolps[i][maxcoords, np.arange(dolps[i].shape[1])]
    longdolps.append(longdolp)

#%%
fig, ax = plt.subplots(3,3)

ax[0,0].plot(longs0s[0])
ax[1,0].plot(longs0s[1])
ax[2,0].plot(longs0s[2])

ax[0,1].plot(longaolps[0])
ax[1,1].plot(longaolps[1])
ax[2,1].plot(longaolps[2])

ax[0,2].plot(longdolps[0])
ax[1,2].plot(longdolps[1])
ax[2,2].plot(longdolps[2])

labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.95, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='black', 
            va='top', 
            ha='left')

ax[0,0].set_ylim([0,7000])
ax[1,0].set_ylim([0,7000])
ax[2,0].set_ylim([0,7000])

ax[0,1].set_ylim([-2,2])
ax[1,1].set_ylim([-2,2])
ax[2,1].set_ylim([-2,2])

ax[0,2].set_ylim([0,1])
ax[1,2].set_ylim([0,1])
ax[2,2].set_ylim([0,1])

ax[0,0].set_ylabel('Metaphase')
ax[1,0].set_ylabel('Anaphase')
ax[2,0].set_ylabel('Telophase')

ax[0,0].set_title(r'$S_0$')
ax[0,1].set_title('AoLP')
ax[0,2].set_title('DoLP')

fig.subplots_adjust(wspace=0.25, hspace=0.65)
fig.supylabel('Phase', x = -0.01)
fig.supxlabel('Perimeter', y = 0.01)
fig.suptitle('Image Mode', y = 1)


#%% normalising images plots

interpolated = NormalisingWidth(data = basic_unwrappeds0, target_length = 2000)
padded = Normaliser(basic_unwrappeds0, 2000)


fig, ax = plt.subplots(3,1, sharex = False)

ax[0].imshow(basic_unwrappeds0, cmap = 'gray')
ax[1].imshow(interpolated, cmap = 'gray')
ax[2].imshow(padded, cmap = 'gray')

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

ax[0].set_title(r'Original $S_0$')
ax[1].set_title(r'Interpolated $S_0$')
ax[2].set_title(r'Padded $S_0$')

fig.subplots_adjust(wspace=0, hspace=-0.5)
fig.supylabel('Thickness', x = 0.05)
fig.supxlabel('Perimeter', y = 0.15)



#%%



#%%
from scipy.interpolate import interp1d

def NormalisingWidth(data, target_length, kind):

    data = np.asanyarray(data)
    current_length = data.shape[-1]
    old_x = np.linspace(0, 1, current_length)
    new_x = np.linspace(0, 1, target_length)
    f = interp1d(old_x, data, kind=kind, axis=-1)
    #try kind = 'cubic' also
    
    return f(new_x)


D1maxcoords = PeakFinderWholeImage(basic_unwrappeds0)
D1longprofile = basic_unwrappeds0[D1maxcoords, np.arange(len(D1maxcoords))]
D1norm_longprofile = NormalisingWidth(data = D1longprofile, target_length = 2000, kind = 'linear')

D1maxcoordsB = PeakFinderWholeImage(basic_unwrappeds0)
D1longprofileB = basic_unwrappeds0[D1maxcoordsB, np.arange(len(D1maxcoordsB))]
D1norm_longprofileB = NormalisingWidth(data = D1longprofileB, target_length = 2000, kind = 'cubic')


D2norm_unwrappeds0B = NormalisingWidth(data = basic_unwrappeds0, target_length = 2000, kind = 'cubic')
D2norm_maxcoordsB = PeakFinderWholeImage(D2norm_unwrappeds0B)
D2norm_longprofileB = D2norm_unwrappeds0B[D2norm_maxcoordsB, np.arange(len(D2norm_maxcoordsB))]

D2norm_unwrappeds0 = NormalisingWidth(data = basic_unwrappeds0, target_length = 2000, kind = 'linear')
D2norm_maxcoords = PeakFinderWholeImage(D2norm_unwrappeds0)
D2norm_longprofile = D2norm_unwrappeds0[D2norm_maxcoords, np.arange(len(D2norm_maxcoords))]

#%% checking difference between 2D and 1D
x = np.arange(len(D2norm_longprofile))
plt.title('Comparison of Kind and Interp')
plt.plot(x, D2norm_longprofileB , label = '2D Cubic')
plt.plot(x, D2norm_longprofile, label = '2D Linear')
plt.plot(x, D1norm_longprofileB, label = '1D Cubic')
plt.plot(x, D1norm_longprofile, label = '1D linear')
plt.legend()

#%%
from scipy.spatial.distance import dice

D1Lin_D1Cub = D1norm_longprofile - D1norm_longprofileB
D1Lin_D2Lin = D1norm_longprofile - D2norm_longprofile
D1Lin_D2Cub = D1norm_longprofile - D2norm_longprofileB
D1Cub_D2Lin = D1norm_longprofileB - D2norm_longprofile
D1Cub_D2Cub = D1norm_longprofileB - D2norm_longprofileB
D2Lin_D2Cub = D2norm_longprofile - D2norm_longprofileB

# plt.plot(x, D1Cub_D2Cub, label = 1, ls = 'dotted')
# plt.plot(x, D1Cub_D2Lin, label = 2, ls = 'dashed')
# plt.plot(x, D1Lin_D1Cub, label = 3, color = 'k')
# plt.plot(x, D1Lin_D2Cub, label = 4, color = 'r')
plt.scatter(x, D1Lin_D2Lin, label = 5, color = 'blue', s = 1)
plt.scatter(x, D2Lin_D2Cub, label = 6, s = 1)
plt.legend()
plt.ylim([-200,200])
#%%
print('1D Lin vs 1D Cub', np.round(np.abs(np.average(D1Lin_D1Cub)), 3))
print('1D Lin vs 2D Lin', np.round(np.abs(np.average(D1Lin_D2Lin)), 3))
print('1D Lin vs 2D Cub', np.round(np.abs(np.average(D1Lin_D2Cub)), 3))
print('1D Cub vs 2D Lin', np.round(np.abs(np.average(D1Cub_D2Lin)), 3))
print('1D Cub vs 2D Cub', np.round(np.abs(np.average(D1Cub_D2Cub)), 3))
print('2D Lin vs 2D Cub', np.round(np.abs(np.average(D2Lin_D2Cub)), 3))

#%%

#%% normalising images



norm1 = Normaliser(datalist = unwrappeds0, target_width = 1700)
norm2 = NormalisingWidth(data = unwrappeds0, target_length = 1700)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))

# ax1.plot(x, pointsaolp)
ax1.imshow(unwrappeds0, cmap = 'gray')
ax1.set_title('Original')

# ax2.plot(x, pointsdolp)
ax2.imshow(norm1, cmap = 'gray')
ax2.set_title('Padding')

# ax3.plot(x, pointss0)
ax3.imshow(norm2, cmap = 'gray')
ax3.set_title('Interpolating')

fig.tight_layout()

#%% normalising lines

# pointsaolp
# pointss0
# pointsdolp

norm11 = Normaliser(datalist = pointss0, target_width = 1700)
norm22 = NormalisingWidth(data = pointss0, target_length = 1700)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))

# ax1.plot(x, pointsaolp)
ax1.plot(pointss0)
ax1.set_title('Original')

# ax2.plot(x, pointsdolp)
ax2.plot(norm11)
ax2.set_title('Padding')

# ax3.plot(x, pointss0)
ax3.plot(norm22)
ax3.set_title('Interpolating')

fig.tight_layout()

#%% normalising lines more

plt.plot(pointss0, label = 'Original')
plt.plot(norm11, label = 'Padding')
plt.plot(norm22, label = 'Interpolating')
plt.legend()
