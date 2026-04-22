#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import Unwrap, BinAveraging, SuperThresholding, \
    ProfileSpecBin, FitGaussian, find_peaks, reader, LocalSegment, set_top, cropper


superimages = reader('..\CellsInterlinked\SuperClear', 'tif')
#vmax=np.percentile(cropped, 95) use this argument whenever doing plt.imshow


#%% Navins Images
navins = superimages[0:5]

for i in range(len(navins)):
    image = navins[i]
    cropped = cropper(image, normalising = True)
    dilated, mask = SuperThresholding(cropped, sigma = 10, dilations = 30, redilations = 50)
    bestimage = dilated * cropped
    
    navinunwrap = Unwrap(image = bestimage, mask = mask, thickness = 300, smoothing = 5)
    navinprofiles = BinAveraging(navinunwrap, bin_size = 2000)
    
    better = set_top(navinunwrap, 99)
    # maxed.append(better)
    plt.figure(i)
    # for j in range(len(navinprofiles)):
        # plt.plot(navinprofiles[j])

    plt.imshow(mask, cmap = 'gray')


#%% Rubys Images
rubys = superimages[5:7]
for i in range(len(rubys)):
    image = rubys[i]
    cropped = cropper(image, normalising = True)
    cropped = set_top(cropped, top = 97.5, zeroes=False)
    dilated, mask = SuperThresholding(cropped, sigma = 10, dilations = 30, redilations = 50)
    bestimage = dilated * cropped
    
    rubyunwrap = Unwrap(image = bestimage, mask = mask, thickness = 300, smoothing = 5)
    rubyprofile = BinAveraging(rubyunwrap, bin_size = 2000)
    
    better = set_top(rubyunwrap, 99)
    
    # plt.figure(i)
    # plt.imshow(mask, cmap = 'gray')
    
    # maxed.append(better)
    plt.figure(i)
    for j in range(len(rubyprofile)):
        plt.plot(rubyprofile[j])
        
#%% Profile plotting
for i in range(len(rubyprofile)):
    fitdata, FWHM = FitGaussian(xaxis, sr_profiles[i], guess = [55000, 150, 20], FWHM = True)
    plt.figure(i)
    plt.plot(sr_profiles[i])
    plt.plot(xaxis, fitdata)
    plt.title(f'Profile {i}, FWHM = {np.round(FWHM, 2)} pix')
    print(FWHM)


#%% The whole lot
#the last image (ruby's telophase) is really sparse but feels like it should work. needs set_top = 95%
images = superimages
for i in range(len(images)-6):
    image = images[-1]
    cropped = cropper(image, normalising = True)
    cropped = set_top(cropped, top = 95.5, zeroes=False)
    dilated, mask = SuperThresholding(cropped, sigma = 10, dilations = 50, redilations = 1)
    # bestimage = dilated * cropped
    
    # rubyunwrap = Unwrap(image = bestimage, mask = mask, thickness = 300, smoothing = 5)
    # rubyprofile = BinAveraging(rubyunwrap, bin_size = 2000)
    
    # better = set_top(rubyunwrap, 99)
    
    plt.figure(i)
    plt.imshow(mask, cmap = 'gray')
    
    # maxed.append(better)
    # plt.figure(i)
    # for j in range(len(rubyprofile)):
        # plt.plot(rubyprofile[j])















