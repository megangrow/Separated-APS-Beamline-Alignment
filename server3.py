import os
import pickle
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("Submit Beam (Step 2)")

# Open data and set parameters
def open_data():
    with open(os.path.join('data', 'all_image_masks.pkl'), 'rb') as f:
        all_image_masks = pickle.load(f)
    # TODO: Switch to getting coordinates from server 2
    with open(os.path.join('data', 'user_coordinates.pkl'), 'rb') as f:
        coords = pickle.load(f)
    x, y = coords['coord1']
    x_coor = int(x)
    y_coor = int(y)
    return all_image_masks, y_coor, x_coor

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

# Save beam edges
def save_edges(edges, min: int, max: int):
    edges.append(min)
    edges.append(max)
    print(f"Beam selected, edges are {max} and {min}")
    with open(os.path.join('data', 'edges.pkl'), 'wb') as f:
        pickle.dump(edges, f)

def calculate_mid(x, y):
    return (x+y)/2

@mcp.tool
def submit_beam() -> str:
    """Submit beam - identifies and saves the horizontal edges of a beam at a given pixel coordinate from image mask data."""
    all_image_masks, y_coor, x_coor = open_data()
    
    # Find and save edge coordinates
    mid, min_x, max_x = get_mid_point(all_image_masks[0], y_coor, x_coor)
    edges = []
    save_edges(edges, min_x, max_x)
    return 'Beam has been submitted'

if __name__ == "__main__":
    #print(submit_beam.fn())
    mcp.run()