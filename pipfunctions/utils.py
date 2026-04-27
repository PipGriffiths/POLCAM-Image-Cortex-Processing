import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
from matplotlib.figure import Figure

from pathlib import Path
import skimage.io
import io
from zipfile import ZipFile


def reader(path: Union[str, Path], 
           imagetype: str) -> List[np.ndarray]:
    '''
    Reads an entire folder of images into a list from a filepath and an image type to search for.
    '''
    images = []
    files = Path(rf'{path}').rglob(rf'*.{imagetype}')
    for file in files:
        images.append(skimage.io.imread(file))
    return images

def figuresaver(imagelist: List[Union[np.ndarray, Figure]], 
                foldername: str, 
                typename: str, 
                formatting: str = 'png', 
                original: bool = False) -> None:
    '''
    Saves images or figures as a zipped folder using a numerical naming system.
    '''
    
    if len(imagelist) == 1:
        content = imagelist[0]
        filename = f'{foldername}.{formatting}'
        
        if original:
            import shutil
            shutil.copy(content, filename)
        else:
            if isinstance(content, plt.Figure):
                content.savefig(filename, format = formatting)
                plt.close(content)
            else:
                fig, ax = plt.subplots()
                ax.imshow(content, cmap='gray')
                ax.set_title(f'{typename} 0')
                plt.savefig(filename, format=formatting)
                plt.close(fig)
        print('Done')
        return
    
    with ZipFile(f"{foldername}.zip", "w") as my_zip:
        
        if original:
            for i, content in enumerate(imagelist):    
                my_zip.write(content, arcname = f'{typename}_{i}.{formatting}')
        
        else:    
            for i, content in enumerate(imagelist):
                
                if isinstance(content, plt.Figure):
                    img_buffer = io.BytesIO()
                    content.savefig(img_buffer, format = 'png')
                    plt.close(content)
                else:    
                    fig, ax = plt.subplots()
                    ax.imshow(content, cmap = 'gray')
                    ax.set_title(f'{typename} {i}')
                    
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format=formatting)
                    plt.close(fig)
                    
                # 4. Write to zip
                my_zip.writestr(f'{typename}_{i}.{formatting}', img_buffer.getvalue())

