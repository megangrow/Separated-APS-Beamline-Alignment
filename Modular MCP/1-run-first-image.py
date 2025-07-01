import os
import glob
import json
import pickle
import torch
import cv2
import numpy as np
import epics as PyEpics
import plotly.express as px
from PIL import Image
from fastmcp import FastMCP
from scipy.ndimage import median_filter
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

mcp = FastMCP("Run First Image (Step 1)")

# For motor testing
class DummyMotor(): 
    def __init__(self, name):
        self.name = name
    def move(self, *args, **kwargs):
        print(f"DummyMotor.move called on {self.name}")

# Parameters
all_image_masks = []
angle_rotation = 5
time_exposure = 2
pixel_size = 1.172
capture_index = 0
file_type = 'TIFF1'
camera_type = 'cam1'
cam_name = '1idPG1'
pname = '/home/mgrow/APS_Test_Images'
froot = 'pin_alignment'
output = '/home/mgrow/Image_Outputs'
test_images = sorted(glob.glob(os.path.join(pname, '*.tiff')))
mtr_samXE   = DummyMotor('1ide1:m34')
mtr_samYE   = DummyMotor('1ide1:m36')
mtr_samOme  = DummyMotor('1ide:m9')
mtr_aeroXE  = DummyMotor('1ide1:m101')

# Set up SAM Model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
mask_generator = SamAutomaticMaskGenerator(sam)

# Create directory to save data
if not os.path.exists('data'):
    os.makedirs('data')

# Move motors and capture images (uses test images)
def move_motor(angle, time_needed):
    global capture_index
    mtr_samOme.move(angle, wait=True)
    if capture_index >= len(test_images): 
        capture_index = 0
    pfname = test_images[capture_index] 
    capture_index += 1
    return pfname

# Move motors and capture images (actual)
# def move_motor(angle, time_needed): 
#     mtr_samOme.move(angle, wait=True)
#     PyEpics.caput(cam_name + ':' + camera_type + ':ImageMode', 'Single', wait=True)
#     PyEpics.caput(cam_name + ':' + camera_type + ':AcquireTime', time_needed, wait=True)
#     PyEpics.caput(cam_name + ':' + file_type +':AutoSave', 'Yes', wait=True)
#     time.sleep(0.05)
#     PyEpics.caput(cam_name + ':' + camera_type + ':Acquire', 1, wait=True)
#     time.sleep(0.05)
#     PyEpics.caput(cam_name + ':' + file_type + ':AutoSave', 'No', wait=True)
#     time.sleep(0.05)
#     fname=PyEpics.caget(cam_name + ':' + file_type + ':FileName_RBV', 'str') + "_%06d"%(PyEpics.caget(cam_name + ':' + file_type + ':FileNumber_RBV')-1) + '.tif'
#     pfname = os.path.join(pname, fname)
#     return pfname

# # Takes two images (w/ and w/out object)
def move_motors_normalize(time_exposure):
    # Move object out of frame + capture
    mtr_samXE.move(-2.0, relative=True, wait=True)
    pfname = move_motor(0, time_exposure)
    image_norm = Image.open(pfname)
    # Move object back to frame + capture
    mtr_samXE.move(2.0, relative=True, wait=True)
    image_path = move_motor(0, time_exposure)
    im = Image.open(image_path)
    width_norm, height_norm = im.size
    print(width_norm, height_norm)
    return width_norm, height_norm, im, image_norm

# Normalizes image and makes the image readable
def normalization(normalize, image1, image0):
    if normalize: 
        first_ch0 = np.array(image0).astype(np.float32) # Convert image to an array
    else: 
        first_ch0 = np.ones(np.array(image0).shape).astype(np.float32) # Convert image to an array of ones
    # Divide image by reference
    image_ch0 = np.array(image1).astype(np.float32)
    epsilon = 1e-6
    norm_image = image_ch0 / (first_ch0 + epsilon)
    # Apply image edits
    norm_image = median_filter(norm_image, 3)
    norm_image = cv2.normalize(norm_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    norm_image = norm_image.astype(np.uint8)
    norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2RGB)
    norm_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)
    return norm_image

# Combine all individual masks into one
def combine_masks(masks):
    merge_mask = np.zeros_like(masks[0]['segmentation'])
    for mask in masks:
        data_mask = mask['segmentation']
        merge_mask += data_mask
        merge_mask = np.clip(merge_mask, 0, 255).astype(np.uint8) 
    return merge_mask

# Save reference image, image masks, normalization answer, height, width, + initial, normalized, and combined mask images
def save_data(im_0, all_image_masks, answer_normalization, width, height, im_1, image_norm, display_mask):
    im_0.save(os.path.join(output, 'reference_image.tiff'))
    with open(os.path.join('data', 'all_image_masks.pkl'), 'wb') as f:
        pickle.dump(all_image_masks, f)
    with open(os.path.join('data', 'normalize_flag.json'), 'w') as f:
        json.dump({'normalize': answer_normalization}, f)
    with open(os.path.join('data', 'image_dimensions.pkl'), 'wb') as f:
        pickle.dump({'width': width, 'height': height}, f)
    fig1 = px.imshow(im_1, color_continuous_scale='gray', template='plotly_dark').update_layout(title="Initial Image")
    fig1.write_image(os.path.join(output, "initial_image.png"))
    fig2 = px.imshow(image_norm, color_continuous_scale='gray', template='plotly_dark').update_layout(title="Normalized Image")
    fig2.write_image(os.path.join(output, "normalized_image.png"))
    cv2.imwrite(os.path.join(output, "normalized_no_grid.png"), image_norm)
    fig3 = px.imshow(display_mask, color_continuous_scale='jet', template='plotly_dark').update_layout(title="Combined Mask")
    fig3.write_image(os.path.join(output, "combined_mask_plot.png"))
    cv2.imwrite(os.path.join(output, "combined_mask.png"), display_mask)
    im_1.save(os.path.join(output, "init_no_grid.png"))

    px.imshow(image_norm, color_continuous_scale='gray', template='plotly_dark').update_layout(title="Normalized Image").show()

@mcp.tool
def run_first_image() -> str:
    """Run first image - captures and processes images for beamline alignment, generating masks and saving data for further steps."""
    global time_exposure
    width, height, im_1, im_0 = move_motors_normalize(time_exposure) # Initial + reference images

    # Normalize image
    answer_normalization = 0 ## TODO: switch this to getting user input
    if answer_normalization:
        image_norm = normalization(1, im_1, im_0)
        answer_normalization = 1
    else:
        image_norm = normalization(0, im_1, im_0)
        answer_normalization = 0

    # Creates segmented masks and combines them into one
    mask_image_norm = mask_generator.generate(image_norm)
    display_mask = combine_masks(mask_image_norm)
    display_mask = (display_mask > 0).astype(np.uint8) * 255
    all_image_masks.append(mask_image_norm)

    save_data(im_0, all_image_masks, answer_normalization, width, height, im_1, image_norm, display_mask)
    return 'First image has been run'

if __name__ == "__main__":
    #print(run_first_image.fn())
    mcp.run()