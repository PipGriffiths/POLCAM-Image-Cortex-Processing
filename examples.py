### Cell 1: Imports
```python
import numpy as np
import matplotlib.pyplot as plt
from ClassPipeline import Pip_PolcamMask, Pip_Unwrap, Pip_Profiles, Pip_Fittings, reader

### Cell 2: Processing the Mask
cell = Pip_PolcamMask(data, offset=150)
cell.Method('polcam')

# Segment using the local thresholding method
cell.Masking(method='local', erosions=5)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cell.s0, cmap='gray')
ax[0].set_title("S0 Intensity")
ax[1].imshow(cell.mask, cmap='jet', alpha=0.5)
ax[1].set_title("Generated Mask")
plt.show()

### Cell 3: Unwrapping and Profiling
# Pass the S0 image and the mask to the unwrapper
unwrapper = Pip_Unwrap(cell.s0, cell.mask, thickness=80)
unwrapped_data = unwrapper.Unwrap().image

# Generate 1D profiles from the straightened image
prof_obj = Pip_Profiles(unwrapped_data, binsize=50)
profiles = prof_obj.Method('sum').profiles

plt.imshow(unwrapped_data, cmap = 'gray')
plt.title("Unwrapped Cell Perimeter")
plt.show()

### Cell 4: Fitting Data
# Fit the first profile using a Gaussian model
fitter = Pip_Fittings(profiles)
fitter.Method(method='gauss')

# Access the first fit result
fit_y = fitter.fit_data[0]
plt.plot(profiles[0], label='Raw Data')
plt.plot(fit_y, label='Gaussian Fit', linestyle='--')
plt.legend()
plt.show()



