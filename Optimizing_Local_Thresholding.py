#%% Preamble 1
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data")
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.io
import numpy as np

#Custom functions
from All_Segmenting_Functions import Polcam, Adaptive_Local_Segment #, Segment_Auto_Local,  \
#        MajorMinor_Axes, Unwrap_General_Contour

polariser_string = '[-45 0; 90 45]'
#%% Preamble 2

tif_files = Path(r'.\CellsInterlinked\RubyCellSet1').rglob('*.tif')
images = []

for file in tif_files:
    images.append(skimage.io.imread(file)) #make a list of images to process

# good easy image, crappy image, difficult image for demo
images_to_do = [images[0], images[4], images[16]]


#%%
for i in range(len(images)):
   
    colors, aolp, dolp, s0 = Polcam(images[i], polariser_string=polariser_string)
    Dilated, Filled, best_block = Adaptive_Local_Segment(s0, 4, 1)


#%% Making Images for one variable change, one image


block_size = [93, 239, 121] #93, 239, 121 best
erosion_list = np.linspace(1,20,20)
dilation_no = 5

dilated_masks = []
filled_masks = []

#%%
for i in range(len(erosion_list)):

    colors, aolp, dolp, s0 = Polcam(images_to_do[0], polariser_string=polariser_string)
    Dilated, Filled = Segment_Auto_Local_Opt(s0, block_size[0], erosion_list[i], dilation_no)
    dilated_masks.append(Dilated)
    filled_masks.append(Filled)

    plt.figure(figsize=(10, 5)) 
    
    plt.subplot(1, 2, 1)
    plt.imshow(dilated_masks[i], cmap='gray')
    plt.title(f'Dilated Mask | Erosion = {erosion_list[i]}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filled_masks[i], cmap='gray')
    plt.title('Filled Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 
  

#%% Making images for one variable change, all images

block_size = [93, 239, 121] #93, 239, 121 best
erosion_list = [i+1 for i in range(20)]
dilation_no = 1

dilated_masks = []
filled_masks = []
mask_blocks = [] 


for j in range(len(images_to_do)):
    colors, aolp, dolp, s0 = Polcam(images_to_do[j], polariser_string=polariser_string)
    
    for i in range(len(erosion_list)):
        Dilated, Filled = Segment_Auto_Local_Opt(s0, block_size[j], erosion_list[i], dilation_no)
        dilated_masks.append(Dilated)
        filled_masks.append(Filled)
        mask_blocks.append(block_size)
        

# %% Plotting the results

total_masks = len(dilated_masks)

for k in range(total_masks):
    num_blocks_per_image = len(erosion_list)
    image_index = k // num_blocks_per_image 
    
    plt.figure(figsize=(10, 5)) 
    
    plt.subplot(1, 2, 1)
    plt.imshow(dilated_masks[k], cmap='gray')
    plt.title(f'Dilated Mask | Img {image_index+1} | Block = {mask_blocks[k]}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(filled_masks[k], cmap='gray')
    plt.title(f'Filled Mask | Img {image_index+1}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 
  
#%% Comments
'''
Round 1 BLOCK SIZES
Least tolerant images is the one of the crappy cell. First tested block_size = np.linspace(101,501,20)
To narrow down a best block size for the most difficult image, test from 150 - 250
The easy example can take a very small block size, 101 is big for it
The telophase example needs 121 currently

Round 2
Tested np.linspace(111, 261, 20)
Most tolerant is the same - easy
Least tolerant needs 245 currently
Telophase needs 121 still

Round 3
Have now tested the telophase images in the np.linspace(111,131,20) range
Can now confirm that the hard telophase image needs block_size = 121

Round 4
Have now tested the difficult image in the np.linspace(229,255,20) range
Can now confirm that it needs block_size = 239

Round 5
For fun, test the np.linspace(11,99,20) on the easy example
Can now confirm that it needs only block_size = 93

Conclusions: block size can vary massively between images
Maybe I want to write some code that tests a range of block sizes
    so that you always get the lowest one to the nearest 10 blocks?
    

Round 6 EROSIONS AND DILATIONS
Skimage.morphology.disk(1) is a r=1 circle on each pixel
    this means it is like [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]]

Keep disk radius = 1 because that is a nice unit circle. 
But there is no reason for that necessarily.


Therefore each erosion is shrinking a footprint of pixels
    to just [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]

For a pixel to remain as 1 (bright) ALL FIVE neighrbouring pixels also
    need to be 1
Otherwise the centre pixel goes to 0

It follows that dilation does almost the opposite
For a pixel to dilate ANY adjacent pixel needs to also be a 1
Therefore it is mathematically the opposite, but in practice the order matters


Round 7
I have tested a variety of erosions using dilation = 5, previously Navin liked erosion = 13, dilation = 5
But not sure how much we should be dilating by at all? 

ASK Ruby why we are eroding and then dilating again?
My unwrapping code already smooths the image so is there any benefit to eroding and dilating?

I would argue no! Which is why the semi-automatic segmenting has looked a little cleaner so far
This only holds because the images are clean and don't have little loops or funny patches

Also if we erode surely we want the same number of dilations again afterwards?
Currently I estimate that 4 erosions is fine and therefore 4 dilations should be also

And it WILL matter because the unwrapping function uses the mask edge

'''

