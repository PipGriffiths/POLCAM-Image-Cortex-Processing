#%%Single Image Script 

#Tasks

# Try the chopping only 50% nearest the peak and fitting the gaussian against that
# Try cubic spline interpolation on the scipy interp1d
# Try two late anaphase cells normalised to the same width and plot them against each other
# Compare a metaphase against this cell

#%%
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np
from GigaFunction import Everything

from Tidy_Functions import Fit_s0, NormalisingWidth, Normaliser, \
    Lorentzianiser, Window, reader, Normaliser, PowerWindow, PeakFinder, \
        ThreshThick, BinAveraging, FitGaussian, GaussianiserAndFit, SigLor

rubys = reader('..\CellsInterlinked\RubyCellSet1', 'tif')
chosen = rubys[8]

size = 100
thick = 50

cdata, keys = Everything(chosen, profiles = True, multprofiles=True, fittings = True,
                         FWHMs = True, thickness=thick, binsize=size)

s0s = cdata.get('s0s')[0]
unwrappeds0s = cdata.get('unwrapped_s0')[0]
maxcoords = cdata.get('max_coords')[0]
profiless0s = cdata.get('profiles_s0')[0]
profilesmult = cdata.get('profiles_mult')[0]
xaxes = cdata.get('x_axes')[0]
fittings = cdata.get('fittings')[0]
fwhms = cdata.get('fwhms')[0]

#%%fittings, mults and profiles

window = PowerWindow(unwrappeds0s, target_max=60, power=1.5)
threshwindow = ThreshThick(window, 31, it = 0, offset = 0)

peaky = PeakFinder(x_data = np.arange(len(profiless0s[14])), y_data = profiless0s[14])
plt.plot(profiless0s[14])
plt.scatter(peaky[0], peaky[1])

#%%
newprofs = BinAveraging(window, bin_size = size)
gennewlor = [Lorentzianiser(i)[0] for i in newprofs]
gennewfwhms = [Lorentzianiser(i)[1] for i in newprofs]

xdata = np.arange(0,50,1)
gennewgau = [FitGaussian(xdata, i) for i in newprofs]
gennewFWHMS = [FitGaussian(xdata, i, FWHM=True)[1] for i in newprofs]


for i in range(len(gennewlor) - 37):
    plt.plot(gennewlor[14])
    plt.plot(profiless0s[14])
print(gennewfwhms)

#%%

fig, ax = plt.subplots(1,1,dpi=300)

for i in range(int(len(profiless0s))):
    # if i == 0:
        # ax.plot(profilesmult[i], color = 'r', label = 'Thresholded')
        # ax.plot(profiless0s[i], color = 'k', label = 'Original')
        # ax.plot(fittings[i], color = 'blue', label = 'Gaussian')
    # ax.plot(profilesmult[i], color = 'r')
    # ax.plot(newprofiles[i], color = 'blue')
    ax.plot(profiless0s[i], color = 'k')
    # ax.plot(newgausses[i], color = 'r', ls = 'dashdot')
    # ax.plot(newlorentzes[i], color = 'k', ls = 'dotted')
    # ax.plot(fittings[i], color = 'blue', ls = 'dashdot')
ax.set_ylabel('Intensity')
ax.set_xlabel('Thickness')
# ax.legend()
ax.set_title('Fittings')


#%% Sigmoid-gaussian method
xdata = np.arange(0,len(profiless0s[0]))

gausses, sigs, fwhmlist = Fit_s0(xdata, profiless0s) #needs thickness = 100 to work. Thicker unwrapping is best

num = 10
plt.imshow(unwrappeds0s, cmap = 'gray')

plt.title(f'SigGauss vs Thresholding Gaussians')
plt.plot(gausses[num], color = 'k', label = 'Gaussian')
# plt.plot(sigs[num], color = 'green', label = 'Sigmoid')
plt.plot(profiless0s[num], color = 'blue', label = 'Original')
plt.plot(fittings[num], color = 'red', label = 'Thresholding Gaussian')
plt.legend()

print(np.max(fwhmlist), np.min(fwhmlist))
print(np.max(fwhms), np.min(fwhms))


#%% plotting fwhms against the length of the unwrapped cell

def forward(x):
    return x*57.5
def inverse(x):
    return x / 57.5
x = np.arange(0, len(unwrappeds0s[0])-(size/2), size) + (size/2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot your data
ax1.plot(x, fwhms)
# ax1.plot(x, newfwhms)
# ax1.plot(x, gennewfwhms)
ax1.scatter(x, fwhms, s = 3)
ax1.set_ylim([0,15])
ax1.set_ylabel('FWHM [pixels]')
secax = ax1.secondary_yaxis('right', functions=(forward, inverse))
secax.set_ylabel('FWHM [nm]')

ax2.imshow(unwrappeds0s, cmap = 'gray', extent=[0, len(unwrappeds0s[0]), 0, len(unwrappeds0s)])
fig.subplots_adjust(hspace = -0.38) 
binlines = [size*i for i in range(len(x))]
for i in range(len(binlines)):
    ax1.axvline(binlines[i], color='r', linestyle='--', linewidth=0.5)
ax2.set_xlabel('Perimeter [pixels]')


