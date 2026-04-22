import numpy as np
import napari
from skimage.filters import sobel
from skimage.graph import route_through_array
from skimage.draw import polygon2mask
from scipy.ndimage import binary_fill_holes
from qtpy.QtWidgets import QApplication
import time

# --- Helper Algorithms ---
def follow_path(image, p1, p2):
    grad = sobel(image)
    cost = 1 / (grad + 1e-6)
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    path, _ = route_through_array(cost, p1, p2, fully_connected=True)
    return np.array(path)

def build_mask(image, points):
    paths = []
    points = np.array(points).astype(int)
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        segment = follow_path(image, p1, p2)
        paths.append(segment[:-1]) 
    contour = np.vstack(paths)
    mask = polygon2mask(image.shape, contour)
    mask = binary_fill_holes(mask)
    return mask, contour

# --- The Main Importable Function ---
def segment_image_interactive(image):
    """
    Opens Napari, allows user to segment, and returns the mask/contour.
    Includes a manual event loop to work in Spyder/Jupyter.
    """
    
    # 1. Local result container
    results = {'mask': None, 'contour': None}

    # 2. Initialize Viewer
    viewer = napari.Viewer()
    viewer.add_image(image, name='cell')
    points_layer = viewer.add_points(ndim=image.ndim, name='clicks', size=5, face_color='red')
    points_layer.mode = 'add'
    # 3. Define the Callback
    def on_enter(viewer):
        points_data = points_layer.data
        
        if len(points_data) < 3:
            print(f"Need at least 3 points. Currently: {len(points_data)}")
            return

        print("Calculating segmentation...")
        
        try:
            # Run Logic
            layer_data = viewer.layers['cell'].data
            mask, contour = build_mask(layer_data, points_data)
            
            # Update results
            results['mask'] = mask
            results['contour'] = contour
            
            print("Segmentation saved internally.")
            print("Closing Napari...")
            viewer.close()
            
        except Exception as e:
            print(f"Error: {e}")

    viewer.bind_key("Enter", on_enter)
    print(">>> Please segment the image and press ENTER <<<")
    
    # 4. Manual Blocking Loop
    # This loop forces Python to wait until the window is closed.
    # processEvents() ensures the GUI doesn't freeze while waiting.
    
    app = QApplication.instance()
    
    if app is None:
        # If no application instance exists (rare in interactive environments)
        # we can't safely proceed without starting a full Qt loop, which we want to avoid.
        raise RuntimeError(
            "Could not find Qt Application instance. Ensure you are running "
            "in an environment (like Spyder or Jupyter) where the Qt backend is enabled."
        )

    # Loop while the results haven't been captured AND the viewer window is still visible.
    while results['mask'] is None and viewer.window.qt_viewer.isVisible():
        # Process GUI events (clicks, draws, etc.) to keep the window responsive.
        app.processEvents() 
        # Sleep briefly to save CPU.
        time.sleep(0.05)
    
    # 5. Check for manual closure before returning
    if results['mask'] is None:
        print("Viewer closed manually without segmentation.")
    
    contour = results['contour']
    contour = contour[:, [1, 0]] # swap y,x coordinates to x,y on the contour
    
    # 5. Return captured results
    return results['mask'], contour

