import os
os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\Project_Python_Scripts")

def reader(path, imagetype):
    from pathlib import Path
    files = list(Path(rf'{path}').rglob(rf'*.{imagetype}'))
    return files

def figuresaver(imagelist, foldername, typename):
    from zipfile import ZipFile
    
    with ZipFile(f"{foldername}.zip", "w") as my_zip:
        for content in imagelist:    
            my_zip.write(content, arcname = content.name)

#%%

rimgs = reader('..\CellsInterlinked\RubyCellSet1', 'tif')

meta_idx = [0, 1, 6, 7, 10, 11, 12, 13, 14, 15]
ana_idx = [2, 3, 4, 5, 8, 9]
telo_idx = [16, 17, 18]

rm = [rimgs[i] for i in meta_idx]
ra = [rimgs[i] for i in ana_idx]
rt = [rimgs[i] for i in telo_idx]


#%%

navinimages = reader('..\CellsInterlinked\\NavinImages', 'tif')

chosen_indexes = [4,6,8,10,17,22,24,26,35,37,41,42,45,47,49,51,52,53,56,57,61,65,73,75,78,80,86,88,94,100,112,119,122]
nimgs = [navinimages[i] for i in chosen_indexes]

nmeta_idx = [3, 5, 6, 7, 8, 9, 11, 17, 18, 19, 20, 21, 22, 26, 31, 32]
nana_idx = [0, 1, 2, 4, 10, 12, 13, 15, 23, 25, 28, 29]
ntelo_idx = [14, 16, 24, 27, 30]

nm = [nimgs[i] for i in nmeta_idx]
na = [nimgs[i] for i in nana_idx]
nt = [nimgs[i] for i in ntelo_idx]

#%%

metas = rm + nm
anas = ra + na
telos = rt + nt

al = metas + anas + telos
print(len(al))
#%%

os.chdir(r"C:\Users\griff\Downloads\Uni All\Coding and Data\Project\ThesisImages")

#%%
figuresaver(imagelist = telos, foldername = 'Telophase', typename = 'Cell')#, formatting = 'ome.tif')

#%%

def reader(path, imagetype):
    from pathlib import Path
    import skimage.io
    images = []
    files = Path(rf'{path}').rglob(rf'*.{imagetype}')
    for file in files:
        images.append(skimage.io.imread(file))
    return images

metaphase = reader('..\ThesisImages\Metaphase', 'tif')

