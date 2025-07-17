"""
This script automates the process of aligning a pin (or object) at beamline center 
using motor control, image normalization, segmentation, and circular path fitting.

Key Steps:
1. Calculates required motor adjustments using beam edge and offset data.
2. Moves motors (sample and aerodynamic stages) to align the object with the beam center.
3. Performs image normalization using reference and current images.
4. Generates segmentation masks on the normalized image to locate the object.
5. Extracts the object midpoint and simulates rotation to track its path.
6. Fits the trajectory to a circular model and generates fit visualizations.
7. Saves regression and scatter plot images for verification.

"""
import os
import numpy as np
from fastmcp import FastMCP

from tools import config
from tools.utils import move_motors_normalize, normalization, combine_masks, get_mid_point, graph_scatter

mcp = FastMCP("Run First Image")

def center_pin_core():
    """Automates pin alignment and verification using motor control, image processing, and segmentation."""
    # Calculate movements based on edges and move motors
    beam_center = config.edges[1] - config.edges[0]
    move_transrotax = ((config.offset - (beam_center / 2)) * config.pixel_size) / 1000
    move_sam_x = (config.radius * (np.sin(config.start_theta * np.pi / 180)) * config.pixel_size) / 1000
    move_sam_y = (-config.radius * (np.cos(config.start_theta * np.pi / 180)) * config.pixel_size) / 1000
    config.mtr_samX.move(move_sam_x, relative=True, wait=True)
    config.mtr_samZ.move(move_sam_y, relative=True, wait=True)
    config.mtr_transRotAx.move(move_transrotax, relative=True, wait=True)

    # Initial + reference images + normalization
    image_norm = normalization(config.answer_normalization, config.im_1, config.im_0)

    # Generate masks from normalized image
    mask_im_norm = config.mask_generator.generate(image_norm)
    display_mask = combine_masks(mask_im_norm)

    # Intial midpoint + edge positions
    x, y = config.object_pt
    first_midpoint, no_x, no_y = get_mid_point(mask_im_norm, y, x)
    
    # Move motor through rotation, generate masks for a combined mask image, fit scatter plot and params
    reg, scatter, params = graph_scatter(first_midpoint, config.angle_rotation, y, config.im_0, config.image_dimensions['width'], config.image_dimensions['height'])
    reg.write_image(os.path.join(config.output, "reg_image.png"))
    scatter.write_image(os.path.join(config.output, "scatterplot_fit.png"))
    return 'Pin Centered and verified'

@mcp.tool
def center_pin() -> str:
    return center_pin_core()