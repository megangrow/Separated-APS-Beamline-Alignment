"""
This module contains core utility functions for the four tools used by main_server.py.
"""
import os
import cv2
import time
import math
import numpy as np
import pandas as pd
import epics as PyEpics
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from tools import config

# Move motors and capture images (uses test images)
def move_motor(angle, time_needed):
    config.mtr_vertRot.move(angle, wait=True)
    if config.capture_index >= len(config.test_images): 
        config.capture_index = 0
    pfname = config.test_images[config.capture_index] 
    config.capture_index += 1
    return pfname

# Move motors and capture images (actual)
# def move_motor(angle, time_needed): 
#     config.mtr_vertRot.move(angle, wait=True)
#     PyEpics.caput(config.cam_name + ':' + config.camera_type + ':ImageMode', 'Single', wait=True)
#     PyEpics.caput(config.cam_name + ':' + config.camera_type + ':AcquireTime', time_needed, wait=True)
#     PyEpics.caput(config.cam_name + ':' + config.file_type +':AutoSave', 'Yes', wait=True)
#     time.sleep(0.05)
#     PyEpics.caput(config.cam_name + ':' + config.camera_type + ':Acquire', 1, wait=True)
#     time.sleep(0.05)
#     PyEpics.caput(config.cam_name + ':' + config.file_type + ':AutoSave', 'No', wait=True)
#     time.sleep(0.05)
#     fname=PyEpics.caget(config.cam_name + ':' + config.file_type + ':FileName_RBV', 'str') + "_%06d"%(PyEpics.caget(config.cam_name + ':' + config.file_type + ':FileNumber_RBV')-1) + '.tif'
#     pfname = os.path.join(config.name, fname)
#     return pfname

# Takes two images (w/ and w/out object)
def move_motors_normalize():
    # Move object out of frame + capture
    config.mtr_samX.move(-2.0, relative=True, wait=True)
    pfname = move_motor(0, config.time_exposure)
    image_norm = Image.open(pfname)
    # Move object back to frame + capture
    config.mtr_samX.move(2.0, relative=True, wait=True)
    image_path = move_motor(0, config.time_exposure)
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
    mask_gen_pts = config.SamAutomaticMaskGenerator(config.sam, points_per_side = None, point_grids = points)
    masks_all = mask_gen_pts.generate(norm_img)

    # See how close detected edges are to known edges
    mp, left, right = get_mid_point(masks_all, y_coor, mid_mask)
    diff_right = calculate_difference(right, config.edges[1])
    diff_left = calculate_difference(left, config.edges[0])
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
        image_file = move_motor(i * config.angle_rotation, config.time_exposure)
        im = Image.open(image_file)
        norm_im = normalization(config.answer_normalization, im, im_0)
        # Find how close edges are
        dif_right, dif_left, mid_point = generate_sam_and_find_edge(mid_for_mask, y_coor, width, height, config.edges, norm_im)
        # If too close, reverse direction and repeat
        if dif_left < 50 or dif_right < 50:
            for j in range(1, int(num_rotations)):
                im_file_rev = move_motor(-j * config.angle_rotation, config.time_exposure)
                im_rev = Image.open(im_file_rev)
                norm_im_rev = normalization(config.answer_normalization, im_rev, im_0)

                dif_right_rev, dif_left_rev, mid_rev = generate_sam_and_find_edge(mid_for_reverse, y_coor, width, height, config.edges, norm_im_rev)
                # If too close on other side, break and go back to original direction
                if dif_left_rev < 50 or dif_right_rev < 50: 
                    break
                else: # Update array and midpoint
                    th_rev -= config.angle_rotation
                    mid_for_reverse = mid_rev
                    theta = np.append(theta, th)
                    coords = np.append(coords, mid_for_reverse)
                    print(f'Midpoint for reverse {mid_for_reverse}')
            break
        else: # Update array and midpoint
            th += config.angle_rotation
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
    display = combine_masks(config.all_image_masks[0])
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