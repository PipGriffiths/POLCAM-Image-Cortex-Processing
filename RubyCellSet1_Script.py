#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import reader, figuresaver, Normaliser
from GigaFunction import Everything

polariser_string = '[-45 0; 90 45]'
polcamimages = reader('..\CellsInterlinked\RubyCellSet1', 'tif')

data, keys = Everything(polcamimages, masks=True)

#%%
print(data.keys())
s0s = data.get('s0s')
unwrappeds0s = data.get('unwrapped_s0')
unwrappedaolp = data.get('unwrapped_aolp')
unwrappeddolp = data.get('unwrapped_dolp')
maxcoords = data.get('max_coords')
xaxes = data.get('x_axes')
masks = data.get('masks')

meta_idx = [0, 1, 6, 7, 10, 11, 12, 13, 14, 15]
ana_idx = [2, 3, 4, 5, 8, 9]
telo_idx = [16, 17, 18]

metas = [unwrappeds0s[i] for i in meta_idx]
anas = [unwrappeds0s[i] for i in ana_idx]
telos = [unwrappeds0s[i] for i in telo_idx]

#%% #average aolp for a bin - single value - with an error - and fit a curve for the average aolp for each 

plt.imshow(s0s[9], cmap = 'gray')
#%%


from Tidy_Functions import BinAveraging

allaverages = []
allstds = []

for i in range(len(unwrappedaolp)):
    averages = []
    stds = []
    profilesaolp = BinAveraging(image = unwrappeddolp[i], bin_size = 50)
    for j in range(len(profilesaolp)):    
        average = np.mean(profilesaolp[j])
        averages.append(average)
        std = np.std(profilesaolp[j])
        stds.append(std)
    allaverages.append(averages)
    allstds.append(stds)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))

ax1.imshow(unwrappeds0s[9], cmap = 'gray')

ax2.imshow(unwrappeddolp[9], cmap = 'gray')
ax2.set_title('Image 9')

xlen = np.arange(len(allaverages[9]))*50
xlen = xlen + 25
ax3.errorbar(x = xlen, y = allaverages[9], yerr = allstds[9], label = f'{i}')

ax3.set_xlabel('Bin Number')
ax3.set_ylabel('AoLP')
ax3.set_ylim([0,1])
fig.tight_layout()


#%% one way of aolp hists
#this time using the coordinates of the max of each column and plotting it as a line profile

figs = []

metaaolps = []
metas0s = []

anaaolps = []
anas0s = []

teloaolps = []
telos0s = []

for i in range(len(maxcoords)):
    
    # aolpimage = unwrappedaolps[i]
    aolpimage = unwrappeddolps[i]
    s0img = unwrappeds0s[i]
    x = np.arange(len(aolpimage[0]))
    
    pointsaolp = []
    pointss0 = []
    num = maxcoords[i]
     
    
    for j in range(len(num)):
        coord = num[j]
        pointaolp = aolpimage[coord]
        pointsaolp.append(pointaolp)
        points0 = s0img[coord]
        pointss0.append(points0)
    
    if i in meta_idx:
        phase = 'Metaphase'
        metaaolps.append(pointsaolp)
        metas0s.append(pointss0)
    elif i in ana_idx:
        phase = 'Anaphase'
        anaaolps.append(pointsaolp)
        anas0s.append(pointss0)
    elif i in telo_idx:
        phase = 'Telophase'
        teloaolps.append(pointsaolp)
        telos0s.append(pointss0)
    else:
        phase = 'Unknown'
    
    fig, (ax1, ax2) = plt.subplots(2, 1)#, figsize=(8, 6))
    
    ax1.plot(x, pointsaolp, label='AoLP')
    ax1.set_title(f'AoLP {phase}, Image No.{i}')
    
    ax2.plot(x, pointss0, label='s0')
    ax2.set_title(f's0 {phase}, Image No.{i}')
    
    fig.tight_layout()
    
    figs.append(fig)

#%% average aolp and s0 by cell phase 

metaaolps = Normaliser(metaaolps)
anaaolps = Normaliser(anaaolps)
teloaolps = Normaliser(teloaolps)

metas0s = Normaliser(metas0s)
anas0s = Normaliser(anas0s)
telos0s = Normaliser(telos0s)

metaaolp_mean = np.nanmean(metaaolps, axis = 0)
anaaolp_mean = np.nanmean(anaaolps, axis = 0)
teloaolp_mean = np.nanmean(teloaolps, axis = 0)

metas0_mean = np.nanmean(metas0s, axis = 0)
anas0_mean = np.nanmean(anas0s, axis = 0)
telos0_mean = np.nanmean(telos0s, axis = 0)


aolpmeans = [metaaolp_mean, anaaolp_mean, teloaolp_mean]
s0means = [metas0_mean, anas0_mean, telos0_mean]

for i in range(3):
    
    if i == 0:
        phase = 'Metaphase'
        x = np.arange(len(aolpmeans[0]))
    elif i == 1:
        phase = 'Anaphase'
        x = np.arange(len(aolpmeans[1]))
    elif i == 2:
        phase = 'Telophase'
        x = np.arange(len(aolpmeans[2]))
    
    fig, (ax1, ax2) = plt.subplots(2, 1)#, figsize=(8, 6))
    
    ax1.plot(x, aolpmeans[i], label='DoLP')
    ax1.set_title(f'Mean AoLP {phase}')
    
    ax2.plot(x, s0means[i], label='s0')
    ax2.set_title(f'Mean s0 {phase}')
    fig.tight_layout() 
    



# figuresaver(imagelist = figs, foldername = 'AoLP_LongProfiles_CellSet1', typename = 'Cell')






















