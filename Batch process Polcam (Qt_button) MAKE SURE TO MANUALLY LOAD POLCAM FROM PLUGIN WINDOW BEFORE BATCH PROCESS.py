# MAKE SURE AFTER RUNNING TO MANUALLY LOAD PLUGIN PLS
import napari
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from napari import Viewer
import os, tifffile
from qtpy.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
#---------------------------------------------------------------------------------------

# Function to save all layers in the viewer
def save_all_layers(viewer, folder_path, image_file_path):
    """
    Save all layers in the viewer as TIFF files in a folder named after the image filename, inside folder_path(where image located).
    """
    if not viewer.layers:
        print("No layers to save.")
        return

    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(image_file_path))[0]

    # Create results folder using the base filename
    save_dir = os.path.join(folder_path, f"results_{base_name}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving all layers to: {save_dir}")

    # Save each layer - is it okay to use tifffile here? or napari buletins ?
    for layer in viewer.layers:
        if isinstance(layer.data, list):  # e.g., time series
            for i, arr in enumerate(layer.data):
                save_path = os.path.join(save_dir, f"{layer.name}_{i}.tif")
                tifffile.imwrite(save_path, arr)
        else:
            save_path = os.path.join(save_dir, f"{layer.name}.tif")
            tifffile.imwrite(save_path, layer.data)

    print(f"Saved all layers to: {save_dir}")


def run_polcam(func, viewer, base_layer):
    """
    Run a PolCam GUI action and revert to the original base layer.
    """
    func()  # run the plugin function (e.g., quadview)
    if base_layer in viewer.layers:
        viewer.layers.selection.active = base_layer
        print(f"Reverted to base layer: {base_layer.name}")
    else:
        print("Base layer no longer exists; cannot revert selection.")



def run_polcam_workflow(viewer):
    w = viewer.window.dock_widgets['main (napari PolCam)']
    prefixes = ('S0_S0', 'S0_S1', 'S0_S2',
                'S1_S0', 'S1_S1', 'S1_S2',
                'S2_S0', 'S2_S1', 'S2_S2') # for the love of god dont start an image name with these combos
    
    if not viewer.layers:
        print("No layers loaded in Napari!")
        return

    base_layer = viewer.layers.selection.active or viewer.layers[0]
    w.dropdown_unit.setCurrentText("[0 -45; 45 90]") #  polarizer settings, i set the default as second setting, change if nescessary
    print(f"Base layer set to: {base_layer.name}")
    
    for f in [w._on_click_quadview, w._on_click_channels, w._on_click_stokes]:
        run_polcam(f, viewer, base_layer)
    print("quadview, channels and  stokes parameters calculated.")
    for layer in list(viewer.layers)[::-1]:
        if any(layer.name.startswith(p) for p in prefixes): # remove all Sx_Sy duplicate layers becuase stokes estimate bugged
            viewer.layers.remove(layer)
            print(f"Removed layer: {layer.name}")

    for f in [w._on_click_aolp, w._on_click_dolp]:
        run_polcam(f, viewer, base_layer)
    print("AoLP and DoLP calculated.")
    w.dropdown_colmap.setCurrentText('DoLPmap') # set colormap button to DoLPmap
    for f in [w._on_click_btn_calculate_colmap]: # run colormap calculation
        run_polcam(f, viewer, base_layer)

    w.dropdown_colmap.setCurrentText('HSVmap') # set colormap button to HSVmap
    for f in [w._on_click_btn_calculate_colmap]: # run colormap calculation
        run_polcam(f, viewer, base_layer)

    #do we need to run reanalyze colormap? or is the defualt ok?
    print("AoLP map and DoLP map calculated.")
    print("Finished processing first layer.")




# this is old code that im afraid to delete yet lol
def select_and_process_folder(viewer: Viewer):
    """
    Ask the user to select a folder, load the first image in the folder,
    and process it using the existing PolCam workflow.
    Returns the folder path.
    """
    # Ask user to select folder
    folder = QFileDialog.getExistingDirectory(
        None, "Select folder containing images"
    )
    if not folder:
        print("No folder selected.")
        return None

    # List image files (common formats)
    img_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.ome')
    files = [f for f in os.listdir(folder) if f.lower().endswith(img_extensions)]
    
    if not files:
        print(f"No image files found in folder: {folder}")
        return folder

    # Sort files to take the first one (you can modify as needed)
    files.sort()
    first_file = os.path.join(folder, files[0])
    print(f"Loading first image: {first_file}")

    # Load image into Napari
    viewer.open(first_file)

    # Run your existing PolCam workflow on this layer
    run_polcam_workflow(viewer)

    # Return folder path for later reference
    return folder, first_file

####################################################################################################
####################### main function to process all images in subfolders ##########################

def process_all_images_in_subfolders(viewer: Viewer):
    """
    select a main directory. 
    Iteratively process all images
    in all subfolders, run PolCam workflow, and save all layers in
    results_[filename] folder. 
    Folders starting with 'results_' are skipped.
    Progress counter
    """
    # Ask user to select the main directory
    main_folder = QFileDialog.getExistingDirectory(None, "Select main directory containing subfolders")
    if not main_folder:
        print("No folder selected.")
        return

    print(f"Processing main directory: {main_folder}")

    img_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.ome')

    # Gather all image paths first, skipping results_ folders
    all_images = []
    for subdir, _, files in os.walk(main_folder):
        if os.path.basename(subdir).startswith("results_"):
            continue
        img_files = [f for f in files if f.lower().endswith(img_extensions)]
        img_files.sort()
        all_images.extend([os.path.join(subdir, f) for f in img_files])

    total_images = len(all_images)
    if total_images == 0:
        print("No images found in any subfolder.")
        return

    # Process each image with a progress counter
    for idx, image_path in enumerate(all_images, start=1):
        print(f"\nProcessing image {idx} of {total_images}: {image_path}")

        # Clear existing layers in Napari before loading new image
        viewer.layers.clear()

        # Load image into Napari
        viewer.open(image_path)

        # Run PolCam workflow
        run_polcam_workflow(viewer)

        # Save all layers in results_[filename] folder in the same subdir
        folder_path = os.path.dirname(image_path)
        save_all_layers(viewer, folder_path, image_path)

    print("\n Finished processing all images in all subfolders.")






def main():
    viewer = napari.Viewer()

    # ---------------------------- Start of custom widget definition ------------------------
    widget = QWidget()
    layout = QVBoxLayout()


    warning_label = QLabel("WARNING: Please ensure 'napari-polcam' is manually loaded from the Plugins menu.")
    warning_label.setStyleSheet("color: red; font-weight: bold;")
    warning_label.setWordWrap(True)
    layout.addWidget(warning_label)
    layout.addSpacing(10) # spacing seperator 

    # Add the button as before
    btn = QPushButton("Run PolCam Batch Processing")
    layout.addWidget(btn)
    widget.setLayout(layout)
    # -------------------------------- End of custom widget definition ---------------------------------

    # Connect the button to your main function
    btn.clicked.connect(lambda: process_all_images_in_subfolders(viewer))

    # Adding Widget
    viewer.window.add_dock_widget(widget, area='right', name="PolCam Batch Processor")
    
    napari.run()

if __name__ == "__main__":
    main()

