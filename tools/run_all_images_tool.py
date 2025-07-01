"""
This script processes a full sequence of beamline images to analyze object motion 
through simulated rotation. It extracts midpoints from segmentation masks, fits 
the motion path to a circular model, and visualizes the scatter and fitted trajectory.

Key Steps:
1. Retrieves the initial object midpoint from the first segmentation mask.
2. Simulates rotational movement by generating masks at each motor angle.
3. Tracks midpoint positions across the sequence using image geometry.
4. Fits the resulting path to a circular arc and extracts parameters:
   - offset (x shift)
   - radius of motion
   - start angle (theta)
5. Saves a regression image and scatter plot to disk for verification.
"""
import os
from fastmcp import FastMCP

from tools import config
from tools.utils import get_mid_point, graph_scatter

mcp = FastMCP("Run First Image")

def run_all_images_core():
    """Processes a series of images to analyze and track midpoint positions through rotation, using segmentation masks and motor control " \
    "simulation, then fits and visualizes the resulting data."""

    # Get initial midpoint and edges from first mask
    x, y = config.object_pt
    first_midpoint, no_x, no_y = get_mid_point(config.all_image_masks[0], y, x)
    
    # Move motor through rotation, generate masks for a combined mask image, fit scatter plot and params
    reg, scatter, params = graph_scatter(first_midpoint, config.angle_rotation, y, config.im_0, config.image_dimensions['width'], config.image_dimensions['height'])
    
    # Save data
    config.params = params
    config.offset = params[0]
    config.radius = params[1]
    config.start_theta = params[2]
    reg.write_image(os.path.join(config.output, "reg_image.png"))
    scatter.write_image(os.path.join(config.output, "scatterplot.png"))
    return 'All images have been run'

@mcp.tool
def run_all_images() -> str:
    return run_all_images_core()
