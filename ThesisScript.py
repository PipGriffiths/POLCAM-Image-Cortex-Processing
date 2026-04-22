#%% Imports and setup
import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")
import matplotlib.pyplot as plt
import numpy as np

from Tidy_Functions import reader, figuresaver, Normaliser, NormalisingWidth
from GigaFunction import Everything

polariser_string = '[-45 0; 90 45]'
metaphase = reader('..\ThesisImages\Metaphase', 'tif')
anaphase = reader('..\ThesisImages\Anaphase', 'tif')
telophase = reader('..\ThesisImages\Telophase', 'tif')

#%%
mdata, keys = Everything(metaphase, unwrapaolp=True, unwrapdolp=True, unwraps0=True)
adata, keys = Everything(anaphase, unwrapaolp=True, unwrapdolp=True, unwraps0=True)
tdata, keys = Everything(telophase, unwrapaolp=True, unwrapdolp=True, unwraps0=True)

#%% extract data
maolps = mdata.get('unwrapped_aolp')
aaolps = adata.get('unwrapped_aolp')
taolps = tdata.get('unwrapped_aolp')
aolps = [maolps, aaolps, taolps]


mdolps = mdata.get('unwrapped_dolp')
adolps = adata.get('unwrapped_dolp')
tdolps = tdata.get('unwrapped_dolp')
dolps = [mdolps, adolps, tdolps]


mcoords = mdata.get('max_coords')
acoords = adata.get('max_coords')
tcoords = tdata.get('max_coords')
coords = [mcoords, acoords, tcoords]


ms0s = mdata.get('unwrapped_s0')
as0s = adata.get('unwrapped_s0')
ts0s = tdata.get('unwrapped_s0')
s0s = [ms0s, as0s, ts0s]

thelot = maolps + aaolps + taolps

#%% averages for a bin
from Tidy_Functions import BinAveraging

aallaverages = []
aallstds = []
dallaverages = []
dallstds = []

for k in range(3):
    aolp = aolps[k]
    dolp = dolps[k]
    for i in range(len(aolp)):
        
        aaverages = []
        astds = []
        daverages = []
        dstds = []
        
        profilesaolp = BinAveraging(image = aolp[i], bin_size = 100)
        profilesdolp = BinAveraging(image = dolp[i], bin_size = 100)
        for j in range(len(profilesaolp)):    
            average = np.mean(profilesaolp[j])
            average = np.mean(profilesdolp[j])
            
            aaverages.append(average)
            daverages.append(average)
            astd = np.std(profilesaolp[j])
            dstd = np.std(profilesaolp[j])
            
            astds.append(astd)
            dstds.append(dstd)
            
        aallaverages.append(aaverages)
        aallstds.append(astds)
        
        dallaverages.append(daverages)
        dallstds.append(dstds)
    


for i in range(len(aallaverages)):
    # fig, (ax1, ax2) = plt.subplots(2, 1)#, figsize=(8, 6))
    
    # ax1.imshow(allaolps[i], cmap = 'gray')
    # ax1.set_title(f'Image {i}')
    
    xlen = np.arange(len(aallaverages[i]))*100
    xlen = xlen + 50
    # ax2.errorbar(x = xlen, y = aallaverages[i], yerr = aallstds[i], label = f'{i}')

    # ax2.set_xlabel('Bin Number')
    # ax2.set_ylabel('AoLP')
    # ax2.set_ylim([-np.pi/2,np.pi/2])
    # fig.tight_layout()


#%% one way of aolp hists
#this time using the coordinates of the max of each column and plotting it as a line profile
#getting the raw long profiles

def LongProfiles(coordinates, phases):
    all_lines = []
    
    for i in range(len(phases)):
        maxcoords = coordinates[i]
        imagelist = phases[i]
        phase_lines = []
        
        for j in range(len(maxcoords)):
            image = imagelist[j]
            num = maxcoords[j]
            points = [image[coord] for coord in num]
            phase_lines.append(points)
        
        all_lines.append(phase_lines)
    
    return all_lines



alls0s = LongProfiles(coords, s0s)
alldolps = LongProfiles(coordinates = coords, phases = dolps)
allaolps = LongProfiles(coords, phases = aolps)

#%% plotting all of the long profiles raw

for k in range(len(alls0s)): 
    if k == 0:
        phase = 'Metaphase'
    elif k == 1:
        phase = 'Anaphase'
    elif k == 2:
        phase = 'Telophase'
    
    for i in range(len(alls0s[k])):
        
        x = np.arange(len(alls0s[k][i]))

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))
        
        ax1.plot(x, allaolps[k][i], label='AoLP')
        ax1.set_title(f'AoLP {phase}, Image No.{i}')
        
        ax2.plot(x, alls0s[k][i], label='s0')
        ax2.set_title(f's0 {phase}, Image No.{i}')
        
        ax3.plot(x, alldolps[k][i])
        ax3.set_title(f'DoLP {phase}, Image No.{i}')
        
        fig.tight_layout()


#%% padding and interpolating

def GetMaxes(data):
    return [max(len(line) for line in phase) for phase in data]
def GetMins(data):
    return [min(len(line) for line in phase) for phase in data]    

       
def NormData(all_phases_data, lengths=None):
    actual_maxes = GetMaxes(all_phases_data)

    if lengths is None:
        lengths = actual_maxes
        lengths = [2006,2006,2006]
        
    padded_out = []
    interp_out = []
    
    can_pad = all(lengths[i] >= actual_maxes[i] for i in range(len(lengths)))

    for i in range(len(all_phases_data)):
        phase_lines = all_phases_data[i]
        target = lengths[i]
        
        i_phase = [NormalisingWidth(line, target) for line in phase_lines]
        interp_out.append(i_phase)
        
        if can_pad:
            p_phase = [Normaliser(line, target) for line in phase_lines]
            padded_out.append(p_phase)
        else:
            padded_out.append(None)

    return padded_out, interp_out


alls0s_padded, alls0s_interp = NormData(all_phases_data = alls0s)
allaolps_padded, allaolps_interp = NormData(all_phases_data = allaolps)
alldolps_padded, alldolps_interp = NormData(all_phases_data = alldolps)

a, shorts0s_interp = NormData(all_phases_data = alls0s, lengths = GetMins(alls0s))
a, shortaolps_interp = NormData(all_phases_data = allaolps, lengths = GetMins(alls0s))
a, shortdolps_interp = NormData(all_phases_data = alldolps, lengths = GetMins(alls0s))   

#%% averaging 
     
def MeanProfile(data):
    meandata = [np.nanmean(r, axis = 0) for r in data]
    return meandata
    
aolpmean_padded = MeanProfile(allaolps_padded)
dolpmean_padded = MeanProfile(alldolps_padded)
s0mean_padded = MeanProfile(alls0s_padded)

aolpmean_interp = MeanProfile(allaolps_interp)
dolpmean_interp = MeanProfile(alldolps_interp)
s0mean_interp = MeanProfile(alls0s_interp)

shortaolpmean = MeanProfile(shortaolps_interp)
shortdolpmean = MeanProfile(shortdolps_interp)
shorts0mean = MeanProfile(shorts0s_interp)
      
#%% plotting interpolated up to longest

for i in range(3):
    
    if i == 0:
        phase = 'Metaphase'
        x = np.arange(len(aolpmean_interp[0]))
    elif i == 1:
        phase = 'Anaphase'
        x = np.arange(len(aolpmean_interp[1]))
    elif i == 2:
        phase = 'Telophase'
        x = np.arange(len(aolpmean_interp[2]))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))
    fig.suptitle('Interpolated')
    
    ax1.plot(x, aolpmean_interp[i], label='DoLP')
    ax1.set_title(f'Mean AoLP {phase}')
    
    ax2.plot(x, s0mean_interp[i], label='s0')
    ax2.set_title(f'Mean s0 {phase}')

    ax3.plot(x, dolpmean_interp[i], label='s0')
    ax3.set_title(f'Mean DoLP {phase}')

    fig.tight_layout() 


#%% plotting interpolated down to shortest

for i in range(3):
    
    if i == 0:
        phase = 'Metaphase'
        x = np.arange(len(shortaolpmean[0]))
    elif i == 1:
        phase = 'Anaphase'
        x = np.arange(len(shortaolpmean[1]))
    elif i == 2:
        phase = 'Telophase'
        x = np.arange(len(shortaolpmean[2]))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, figsize=(8, 6))
    fig.suptitle('Interpolated')
    
    ax1.plot(x, shortaolpmean[i], label='DoLP')
    ax1.set_title(f'Mean AoLP {phase}')
    
    ax2.plot(x, shorts0mean[i], label='s0')
    ax2.set_title(f'Mean s0 {phase}')

    ax3.plot(x, shortdolpmean[i], label='s0')
    ax3.set_title(f'Mean DoLP {phase}')

    fig.tight_layout() 

#%% comparing two methods for each phase AoLP

for i in range(len(aolpmean_interp)):
    
    if i == 0:
        phase = 'Metaphase'
        x = np.arange(len(aolpmean_interp[0]))
    elif i == 1:
        phase = 'Anaphase'
        x = np.arange(len(aolpmean_interp[1]))
    elif i == 2:
        phase = 'Telophase'
        x = np.arange(len(aolpmean_interp[2]))
    
    fig, (ax1, ax2) = plt.subplots(2,1, dpi = 300)
    fig.suptitle(f'{phase}')
    
    ax1.plot(x, aolpmean_padded[i])
    ax1.set_title('Padded AoLP')
    
    ax2.plot(x, aolpmean_interp[i])
    ax2.set_title('Interpolated AoLP')

    fig.tight_layout()

#%% comparing on a single plot per phase
fig, ax = plt.subplots(3, 1, sharex = True)
fig.suptitle(r'Average Long $S_0$ Profiles per Cell Phase', y = 1.01)

# I've simplified your if/else block using a list
phases = ['Metaphase', 'Anaphase', 'Telophase']

for i in range(3):
    x = np.arange(len(s0mean_interp[i]))

    ax[i].set_title(phases[i])
    
    # Plot as usual
    # ax[i].plot(x, s0mean_padded[i], label='Padded')
    ax[i].plot(x, s0mean_interp[i], label='Interpolated')
    ax[i].set_ylim([0,5000])

handles, labels = ax[0].get_legend_handles_labels()
ax[2].legend(handles, labels, loc='lower center')
fig.supylabel('Intensity')
fig.supxlabel('Perimeter')

labels = ['a)', 'b)', 'c)']
for i, j in enumerate(ax.flatten()):
    # Add the text label
    # x=0.05, y=0.95 places it near the top-left corner
    j.text(0.05, 0.05, labels[i], 
            transform=j.transAxes, 
            fontsize=10, 
            color='black', 
            va='bottom', 
            ha='left')

fig.subplots_adjust(wspace=-0.45, hspace=0.35)