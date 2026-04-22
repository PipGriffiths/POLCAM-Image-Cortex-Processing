import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data")
import matplotlib.pyplot as plt
from pathlib import Path
import skimage.io

#Custom functions
from All_Segmenting_Functions import Polcam,\
    Segment_SemiAuto_Napari, Segment_Auto_Local,  \
        MajorMinor_Axes, Segment_Auto_Global,  \
            Unwrap_General_Contour, Unwrap_NotNorm

polariser_string = '[-45 0; 90 45]'
#%%

tif_files = Path(r'.\CellsInterlinked\RubyCellSet1').rglob('*.tif')
images = []

for file in tif_files:
    images.append(skimage.io.imread(file))

# good easy image, crappy image, difficult image for demo
images_to_do = [images[0], images[4], images[16]]

names_norm = ['Cell1_Anaphase_Unwrapped_Normalised.tif', 'Cell5_Anaphase_Unwrapped_Normalised.tif', 'Cell2_Telophase_Unwrapped_Normalised.tif']
names_notnorm = ['Cell1_Anaphase_Unwrapped.tif', 'Cell5_Anaphase_Unwrapped.tif', 'Cell2_Telophase_Unwrapped.tif']


#%%
for i in range(len(images_to_do)):
    
    picture = images_to_do[i]
    colors, aolp, dolp, s0 = Polcam(picture, polariser_string = polariser_string)
    Dilated_Masks, Filled_Local_Masks = Segment_Auto_Local(s0)
    unwrapped, orig_coords, unwrapped_coords = Unwrap_General_Contour(s0, Filled_Local_Masks, 50, 5)
    unwrapped_notnorm = Unwrap_NotNorm(s0, Filled_Local_Masks, 50, 5)

    # s_coords = unwrapped_coords[0, :]
    # d_coords = unwrapped_coords[1, :]
    # max_s, max_d = s_coords[0], d_coords[0]
    # min_s, min_d = s_coords[1], d_coords[1] #brightest and dimmest points on the cortex (unwrapped image)
    # x_orig = orig_coords[0, :]
    # y_orig = orig_coords[1, :]
    # max_x, max_y = x_orig[0], y_orig[0]
    # min_x, min_y = x_orig[1], y_orig[1] #brightest and dimmest points on the cortex (original image)

    plt.figure((3*i)+1)
    plt.imshow(picture, cmap = 'gray')    
    plt.title('Original')
    plt.axis('off')
    plt.show()
    
    plt.figure((3*i)+2)
    plt.imshow(unwrapped_notnorm, cmap = 'gray')
    plt.title('Unwrapped NotNorm')
    plt.axis('off')
    plt.show()

    
#%%

'''
Have a go at dilated mask unwrapping from the local thresholding 
Stop on the global thresholding
Keep the current unwrapped function!
Make a new one for the dilated mask
Test the high contrast image
Get the unwrapping as clean as possible through maybe some interpolation?
Possibly oversampling? Test doing every 4 pixels or some number!
'''



