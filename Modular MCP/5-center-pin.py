import math
import time
import torch
import json
import os
import pickle
import cv2
import glob
import numpy as np
import pandas as pd
import epics as PyEpics
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
from fastmcp import FastMCP
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

mcp = FastMCP("Center Pin and Verify (Step 4)")

# For motor testing
class DummyMotor():
    def __init__(self, name):
        self.name = name
    def move(self, *args, **kwargs):
        print(f"DummyMotor.move called on {self.name}")

# Open data and set parameters
def open_data():
    with open(os.path.join('data', 'edges.pkl'), 'rb') as f:
        edges = pickle.load(f)
    with open(os.path.join('data', 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    with open(os.path.join('data', 'coordinates.pkl'), 'rb') as f:
        x_universal, y_universal = pickle.load(f)
    with open(os.path.join('data', 'all_image_masks.pkl'), 'rb') as f:
        all_image_masks = pickle.load(f)
    with open(os.path.join('data', 'normalize_flag.json'), 'r') as f:
        data = json.load(f)
    answer_normalization = data['normalize']
    return edges, params, x_universal, y_universal, all_image_masks, answer_normalization

edges, params, x_universal, y_universal, all_image_masks, answer_normalization = open_data()
angle_rotation = 5
time_exposure = 2
pixel_size = 1.172
capture_index = 0
offset = params[0]
radius = params[1]
start_theta = params[2]
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

# Set up SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
mask_generator = SamAutomaticMaskGenerator(sam)

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

# Takes two images (w/ and w/out object)
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

# Finds edge coordinates by checking each mask
def get_mid_point(masks, y, x):
    pixel = None
    for i, mask in enumerate(masks): # Loop through all masks to find one with coords
        segmentation_mask = mask['segmentation']
        if segmentation_mask[y,x]:
            pixel = i
        else:
            pass
    row_mask = masks[pixel]['segmentation'][y,:] # Extract horizontal row
    object_indices = np.where(row_mask)[0]
    if len(object_indices) > 0: # Calculate midpoint and mask edges
        min_x = np.min(object_indices)
        max_x = np.max(object_indices)
        midpoint = calculate_mid(min_x, max_x)
        return int(midpoint), int(min_x), int(max_x)
    else:
        return "Issue calculating midpoint and/or edges"

# Find key points to create masks and checks distance from midpoint to edge
def generate_sam_and_find_edge(mid_mask, y_coor, w, h, input_edges, norm_img):
    # Calculate edges of midpoint + mask
    x_grid, y_grid = calculate_point(mid_mask, y_coor, w, h)
    x_grid_edge, y_grid_edge = calculate_point(input_edges[0], input_edges[1], w, h)

    # Pass along points of interest to the generator to focus on
    points = [np.array([[x_grid, y_grid], [x_grid_edge, y_grid_edge]])]
    mask_gen_pts = SamAutomaticMaskGenerator(sam, points_per_side = None, point_grids = points)
    masks_all = mask_gen_pts.generate(norm_img)

    # See how close detected edges are to known edges
    mp, left, right = get_mid_point(masks_all, y_coor, mid_mask)
    diff_right = calculate_difference(right, edges[1])
    diff_left = calculate_difference(left, edges[0])
    return diff_right, diff_left, mp

# Rotate a motor + take images, if the SAM mask midpoint get too close, reverse
def graph_scatter(first_midpoint, rots, y_coor, im_0, width, height):
    coords = np.array([])
    theta = np.array([])
    midpoint = first_midpoint
    mid_for_reverse = first_midpoint
    mid_for_mask = first_midpoint
    coords = np.append(coords, midpoint)
    theta = np.append(theta, 0)
    th = 0
    th_rev = 0
    num_rotations = 180/rots

    # Loop through and take images at each rotation angle
    for i in range(1, int(num_rotations)): 
        image_file = move_motor(i * angle_rotation, time_exposure)
        im = Image.open(image_file)
        if answer_normalization:
            norm_im = normalization(1, im, im_0)
        else:
            norm_im = normalization(0, im, im_0)
        # Find how close edges are
        dif_right, dif_left, mid_point = generate_sam_and_find_edge(mid_for_mask, y_coor, width, height, edges, norm_im)
        # If too close, reverse direction and repeat
        if dif_left < 50 or dif_right < 50:
            for j in range(1, int(num_rotations)):
                im_file_rev = move_motor(-j * angle_rotation, time_exposure)
                im_rev = Image.open(im_file_rev)
                if answer_normalization:
                    norm_im_rev = normalization(1, im_rev, im_0)
                else:
                    norm_im_rev = normalization(0, im_rev, im_0)

                dif_right_rev, dif_left_rev, mid_rev = generate_sam_and_find_edge(mid_for_reverse, y_coor, width, height, edges, norm_im_rev)
                # If too close on other side, break and go back to original direction
                if dif_left_rev < 50 or dif_right_rev < 50: 
                    break
                else: # Update array and midpoint
                    th_rev -= angle_rotation
                    mid_for_reverse = mid_rev
                    theta = np.append(theta, th)
                    coords = np.append(coords, mid_for_reverse)
                    print(f'Midpoint for reverse {mid_for_reverse}')
            break
        else: # Update array and midpoint
            th += angle_rotation
            theta = np.append(theta, th)
            mid_for_mask = mid_point
            print(f'Midpoint is now {mid_for_mask}')
            coords = np.append(coords, mid_for_mask)

    # Create plots
    print(f'Coords: {coords}, Theta: {theta}')
    max_ = np.max(coords)
    min_ = np.min(coords)
    rad = (max_ - min_) / 2
    off = (max_ + min_) / 2
    df = pd.DataFrame({'x': theta, 'y': coords})
    bounds = ([-3000, 0, -180], [3000, 2000, 180])
    p0 = [off, rad, 0]
    params, params_cov = curve_fit(func_fitting, theta, coords, p0=p0, bounds=bounds)
    df_fit = pd.DataFrame({'x': theta, 'y': [math.ceil(value) for value in func_fitting(theta, params[0], params[1], params[2])]})
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(x=df['x'], y=df['y'], name="Measured Midpoints"))
    scatter.add_trace(go.Scatter(x=df_fit['x'], y=df_fit['y'], name="Fitted Curve"))
    scatter.update_layout(xaxis_title='Angle (Degrees)', yaxis_title='Midpoint', plot_bgcolor='black')
    scatter.update_traces(mode='markers')
    display = combine_masks(all_image_masks[0])
    reg = px.imshow(display, color_continuous_scale='gray')
    return reg, scatter, params

def calculate_mid(x, y):
    return (x+y)/2

def calculate_difference(x, y):
    return abs(x-y)

def calculate_point(x, y, im_size_x, im_size_y):
    px = x/im_size_x
    py = y/im_size_y
    return px, py

def func_fitting(x, oft, rad, st):
    return oft+rad*np.sin((st+x)*np.pi/180)

@mcp.tool
def center_pin_and_verify() -> str:
    """Center Pin and Verify - Automates pin alignment and verification using motor control, image processing, and segmentation."""
    # Calculate movements based on edges and move motors
    beam_center = edges[1] - edges[0]
    move_aero_x = ((offset - (beam_center / 2)) * pixel_size) / 1000
    move_sam_x = (radius * (np.sin(start_theta * np.pi / 180)) * pixel_size) / 1000
    move_sam_y = (-radius * (np.cos(start_theta * np.pi / 180)) * pixel_size) / 1000
    mtr_samXE.move(move_sam_x, relative=True, wait=True)
    mtr_samYE.move(move_sam_y, relative=True, wait=True)
    mtr_aeroXE.move(move_aero_x, relative=True, wait=True)

    # Initial + reference images + normalization
    width, height, im_1, im_0 = move_motors_normalize(time_exposure)
    if answer_normalization:
        image_norm = normalization(1, im_1, im_0)
    else:
        image_norm = normalization(0, im_1, im_0)

    # Generate masks from normalized image
    mask_im_norm = mask_generator.generate(image_norm)
    display_mask = combine_masks(mask_im_norm)

    # Intial midpoint + edge positions
    first_midpoint, no_x, no_y = get_mid_point(mask_im_norm, y_universal, x_universal)
    
    # Move motor through rotation, generate masks for a combined mask image, fit scatter plot and params
    reg, scatter, params = graph_scatter(first_midpoint, angle_rotation, y_universal, im_0, width, height)
    sam_img = px.imshow(display_mask, color_continuous_scale='gray', template='plotly_dark')
    sam_img.write_image(os.path.join(output, "sam_image.png"))
    scatter.write_image(os.path.join(output, "scatter_plot5.png"))
    return 'Pin Centered and verified'

if __name__ == "__main__":
    #mcp.run()
    print(center_pin_and_verify.fn())