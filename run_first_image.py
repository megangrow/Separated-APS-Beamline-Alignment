########################################################################################
# This code consists of the first 'step' of beamline_alignment_with_motor_code.py
# This means the steps that are executed when the user hits 'Run First Image' on the GUI
# Adapted from: https://github.com/AdvancedPhotonSource/auto_beamline_alignment_tomo
# Megan Grow, Argonne DSL SULI Intern - 06/06/2025
########################################################################################
import os
import glob
import json
import pickle
import plotly.express as px
import numpy as np
from PIL import Image
import epics as PyEpics
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
import cv2
from scipy.ndimage import median_filter

# Print intro
print(f'***********************************************************************************************************************************')
print(f'Three plots will open in your browser. Please pick a coordinate inside the beam, outside the object for entering into Step 2 \'submit_beam.py\'')
print(f'Please pick another coordinate inside the object for entering into Step 3 \'run_all_images.py\'')
print(f'***********************************************************************************************************************************')

# This class creates a dummy motor for testing purposes
class DummyMotor():
    def __init__(self, name):
        self.name = name
    def move(self, *args, **kwargs):
        print(f"DummyMotor.move called on {self.name}")

# Set parameters
all_image_masks = []
pname = '/home/mgrow/APS_Test_Images' # This will change based on local setup
time_exposure = 2
angle_rotation = 5
samXE_pv = '1ide1:m34'
samYE_pv = '1ide1:m36'
aeroXE_pv = '1ide1:m101'
aero_pv = '1ide:m9'
pixel_size = 1.172
file_type = 'TIFF1'
camera_type = 'cam1'
cam_name = '1idPG1'
capture_index = 0 # For dummy image aquisition
froot = 'pin_alignment'
test_images = sorted(glob.glob(os.path.join(pname, '*.tiff')))
output = '/home/mgrow/Image_Outputs' # This will change based on local setup

# Get user input for normalization
answer_normalization = input('Would you like to normalize the image (yes/no): ').strip().lower() in ['yes', 'y', 'true', '1']

# Set up motors and camera - actual
# mtr_samXE = PyEpics.Motor(samXE_pv)
# mtr_samYE = PyEpics.Motor(samYE_pv)
# mtr_samOme = PyEpics.Motor(aero_pv)
# mtr_aeroXE = PyEpics.Motor(aeroXE_pv)
# PyEpics.caput(cam_name + ':' + file_type + ':FileName', froot, wait=True)

# Set up motors - test
mtr_samXE = DummyMotor(samXE_pv)
mtr_samYE = DummyMotor(samYE_pv)
mtr_samOme = DummyMotor(aero_pv)
mtr_aeroXE = DummyMotor(aeroXE_pv)

# Set up SAM Mask Generator - the sam checkpoint varies on your local setup
sam_checkpoint = 'sam_vit_h_4b8939.pth'
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

########################################################################################
# This function is used to move motors, capture images, and return filename
# Option for actual image acquisition with cameras or using a file of test images
def move_motor(angle, time_needed):
    # Image acquisition - actual
    ############################
    # mtr_samOme.move(angle, wait=True)
    # PyEpics.caput(cam_name + ':' + camera_type + ':ImageMode', 'Single', wait=True)
    # PyEpics.caput(cam_name + ':' + camera_type + ':AcquireTime', time_needed, wait=True)
    # PyEpics.caput(cam_name + ':' + file_type +':AutoSave', 'Yes', wait=True)
    # time.sleep(0.05)
    # PyEpics.caput(cam_name + ':' + camera_type + ':Acquire', 1, wait=True)
    # time.sleep(0.05)
    # PyEpics.caput(cam_name + ':' + file_type + ':AutoSave', 'No', wait=True)
    # time.sleep(0.05)
    # fname=PyEpics.caget(cam_name + ':' + file_type + ':FileName_RBV', 'str') + "_%06d"%(PyEpics.caget(cam_name + ':' + file_type + ':FileNumber_RBV')-1) + '.tif'
    # pfname = os.path.join(pname, fname)
    # return pfname

    # Image acquisition - test w/ APS_Test_Images files
    ####################################################
    global capture_index
    mtr_samOme.move(angle, wait=True)
    if capture_index >= len(test_images):  # Verify image exists
        capture_index = 0
    pfname = test_images[capture_index] # Gets test image at index
    capture_index += 1
    return pfname

########################################################################################
# This function takes an image without object and then moves object to frame and takes another picture
def move_motors_normalize(time_exposure):
    # Moves object out of the way and take a reference image (pfname = filename)
    mtr_samXE.move(-2.0, relative=True, wait=True)
    pfname = move_motor(0, time_exposure)
    image_norm = Image.open(pfname)

    # Move object back into frame
    mtr_samXE.move(2.0, relative=True, wait=True)
    image_path = move_motor(0, time_exposure)

    im = Image.open(image_path)
    width_norm, height_norm = im.size
    return width_norm, height_norm, im, image_norm

########################################################################################
# This function normalizes image (if indicated)
def normalization(normalize, image1, image0):
    if normalize: # Convert to an array
        first_ch0 = np.array(image0).astype(np.float32)
    else: # Convert to an array of ones (will leave image unchanged)
        first_ch0 = np.ones(np.array(image0).shape).astype(np.float32)

    # Divide image by reference
    image_ch0 = np.array(image1).astype(np.float32)
    norm_image = image_ch0 / first_ch0

    # Image edits: filter to reduce noise, normalize (0-255), convert to RGB, apply ColorMap
    norm_image = median_filter(norm_image, 3)
    norm_image = cv2.normalize(norm_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
    norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
    return norm_image

########################################################################################
# This function takes all individual masks and combines them into one for visibility
def combine_masks(masks):
    # Create empty array for masks to be added on
    merge_mask = np.zeros_like(masks[0]['segmentation'])

    # Loop through all masks: extracts current segmentation mask and adds to the merged mask
    for mask in masks:
        data_mask = mask['segmentation']
        merge_mask += data_mask
        merge_mask = np.clip(merge_mask, 0, 255).astype(np.uint8) # Keeps image valid
    return merge_mask

########################################################################################
#   Beginning of main code block - update_imgs() in beamline code
########################################################################################

# Moves motors to take initial + reference images
width, height, im_1, im_0 = move_motors_normalize(time_exposure)
im_0.save(os.path.join(output, 'reference_image.tiff')) # Save for later

# Normalizes the initial image + updates image_norm
if answer_normalization:
    image_norm = normalization(1, im_1, im_0)
    answer_normalization = 1
else:
    image_norm = normalization(0, im_1, im_0)
    answer_normalization = 0

# Creates all of the segmented masks using SAM and combines for visibility
mask_image_norm = mask_generator.generate(image_norm)
display_mask = combine_masks(mask_image_norm)
display_mask = (display_mask > 0).astype(np.uint8) * 255
all_image_masks.append(mask_image_norm)

# Save all_image_masks for use in Step 2
with open('all_image_masks.pkl', 'wb') as f:
    pickle.dump(all_image_masks, f)

# Save normalization answer for Steps 3, 4
with open('normalize_flag.json', 'w') as f:
    json.dump({'normalize': answer_normalization}, f)

# Save width and height for Step 3
with open('image_dimensions.pkl', 'wb') as f:
    pickle.dump({'width': width, 'height': height}, f)

# Create + launch interactive image plots for each image
print("Plots opening in your browser.")
px.imshow(im_1, color_continuous_scale='gray', template='plotly_dark').update_layout(title="Initial Image").show()
px.imshow(image_norm, color_continuous_scale='gray', template='plotly_dark').update_layout(title="Normalized Image").show()
px.imshow(display_mask, color_continuous_scale='jet', template='plotly_dark').update_layout(title="Combined Mask").show()

# Get and save coordinates for Steps 2, 3
x1 = float(input("Enter x for first coordinate (inside beam): "))
y1 = float(input("Enter y for first coordinate (inside beam): "))
x2 = float(input("Enter x for second coordinate (inside object): "))
y2 = float(input("Enter y for second coordinate (inside object): "))
coord1 = (x1, y1)
coord2 = (x2, y2)
with open('user_coordinates.pkl', 'wb') as f:
    pickle.dump({'coord1': coord1, 'coord2': coord2}, f)