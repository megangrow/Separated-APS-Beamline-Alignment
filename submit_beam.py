########################################################################################
# This code consists of the second 'step' of beamline_alignment_with_motor_code.py
# This means the steps that are executed when the user hits 'Submit Beam' on the GUI
# Adapted from: https://github.com/AdvancedPhotonSource/auto_beamline_alignment_tomo
# Megan Grow, Argonne DSL SULI Intern - 06/06/2025
########################################################################################
import numpy as np
import pickle
import os

########################################################################################
# This function finds the edge coordinates by going through all masks to find one 
# containing the x,y coordinates passed in
def get_mid_point(masks, y, x):
    pixel = None
    # Loop through all masks to find one containing object coordinates
    for i, mask in enumerate(masks):
        segmentation_mask = mask['segmentation']
        if segmentation_mask[y,x]:
            pixel = i
        else:
            pass
    
    # Extract a horizontal row at the vertical coordinate + create array
    row_mask = masks[pixel]['segmentation'][y,:]
    object_indices = np.where(row_mask)[0]

    # Calculate and return the midpoint and mask edges
    if len(object_indices) > 0:
        min_x = np.min(object_indices)
        max_x = np.max(object_indices)
        midpoint = calculate_mid(min_x, max_x)
        return int(midpoint), int(min_x), int(max_x)
    else:
        return "Issue calculating midpoint and/or edges"
    
########################################################################################
# Helper functions
def calculate_mid(x, y):
    return (x+y)/2

########################################################################################
#   Beginning of main code block - update_edge() in beamline code
########################################################################################

# Load all image masks from Step 1
with open(os.path.join('data', 'all_image_masks.pkl'), 'rb') as f:
    all_image_masks = pickle.load(f)

# Get x,y coordinates inside beam from Step 1
with open(os.path.join('data', 'user_coordinates.pkl'), 'rb') as f:
    coords = pickle.load(f)
x, y = coords['coord1']
x_coor = int(x)
y_coor = int(y)
edges = []

# Calculate, save, and print edges
mid, min_x, max_x = get_mid_point(all_image_masks[0], y_coor, x_coor)
edges.append(min_x)
edges.append(max_x)
print(f"Beam selected, edges are {max_x} and {min_x}")

# Save edges[] for Step 3
with open(os.path.join('data', 'edges.pkl'), 'wb') as f:
    pickle.dump(edges, f)