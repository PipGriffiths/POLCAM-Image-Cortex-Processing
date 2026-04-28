## POLCAM-Image-Cortex-Processing
A series of python functions for bioimages of the cortex. These functions segment, unwrap and then fit a gaussian for a FWHM to estimate the cortical thickness of the cell. It provides a modular pipeline for calculating Stokes parameters, segmenting cells, unwrapping perimeters, and performing advanced peak fitting. Ultimately, we find the thickness of the cortex at any point along the cell.

Click on main to see the filetree to find the lab diary version of my code The neatest version however, is main and is all classes

Core Modules:
Pip_PolcamMask: Handles background subtraction, Stokes parameter calculation ($S_0$, AoLP, DoLP), and segmentation (Otsu, Adaptive, or Semi-Auto via Napari).
Pip_Unwrap: Transforms circular/elliptical masks into straightened "unwrapped" coordinate systems.
Pip_Profiles: Bins and averages unwrapped data to generate 1D signal intensities.
Pip_Fittings: Advanced peak fitting using Gaussian, Lorentzian, or Sigmoidal models.

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/POLCAM-Image-Cortex-Processing.git](https://github.com/yourusername/POLCAM-Image-Cortex-Processing.git)
   cd POLCAM-Image-Cortex-Processing
2. pip install -r requirements.txt

# Quick Start
import skimage.io as io
from POLCAM-Image-Cortex-Processing import Pip_PolcamMask, Pip_Unwrap, Pip_Profiles, Pip_Fittings, reader, figuresaver
# 1. Load and process raw PolCam data
raw_img = io.imread('cell_data.tif')
pol = Pip_PolcamMask(raw_img, offset=100)
pol.Method('polcam').Masking(method='local')

# 2. Unwrap the cell perimeter
unwrapper = Pip_Unwrap(pol.s0, pol.mask)
unwrapped_img = unwrapper.Unwrap().image

# 3. Generate profiles
profiler = Pip_Profiles(unwrapped_img)
profiles = profiler.Method('sum').profiles



