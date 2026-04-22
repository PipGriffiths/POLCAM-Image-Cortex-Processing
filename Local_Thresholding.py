import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data")
import matplotlib.pyplot as plt
import skimage as ski
import skimage.io
from Better_HSV_function import HSV_maker
from skimage import measure, morphology
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_fill_holes
import skimage.segmentation as ski_seg
import skimage.morphology as ski_morph

#%%

def segmenter(image):
    '''
    Parameters
    ----------
    image : 2D image.

    Returns
    -------
    dilated_mask : Mask to unwrap.
    binary_local : Binary filled mask.
    '''
    
    colors, AoLP, DoLP, S0 = HSV_maker(image, '[-45 0; 90 45]')
    image = S0
    normalised_image = (image - image.min()) / (image.max() - image.min())
    
    image = ski.util.img_as_ubyte(normalised_image)
    
    block_size = 251
    
    local_thresh = ski.filters.threshold_local(image, block_size, offset = 5)
    binary_local = image > local_thresh
    binary_local = ski_seg.clear_border(binary_local)
    
    cleared_mask = ski_morph.remove_small_objects(binary_local, min_size = 250)
    filled_mask = binary_fill_holes(cleared_mask)
    
    selem = ski_morph.disk(1) 
    
    for i in range(13): #up to interpretation
        filled_mask = ski_morph.erosion(filled_mask, selem)
    
    largest_edge = find_boundaries(filled_mask, mode='inner')
    dilated_mask = morphology.dilation(largest_edge, selem)
    
    for  i in range(5): #up to interpretation
        dilated_mask = morphology.dilation(dilated_mask, selem)
    return dilated_mask, filled_mask



#%% Give it all the images

from pathlib import Path


tif_files = Path('.').rglob('*.tif')
images = []

for file in tif_files:
    images.append(skimage.io.imread(file)) #make a list of images to process


for i in range(len(images)):
    plt.figure(i)
    dilation, binar = segmenter(images[i])
    #plt.imshow(images[i], cmap = 'gray')
    plt.imshow(binar, cmap = 'gray') #, alpha = 0.35)
    plt.axis('off')
#%%
'''
Now I want to compare global with local thresholding    - urgent
And compare that against my semi-automatic method also  - urgent

Test loads of block sizes and offsets
'''



