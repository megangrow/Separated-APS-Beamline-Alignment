"""
This script is the first step of a beamline image processing pipeline. It captures an initial and 
reference image, performs normalization, generates segmentation masks using a SAM-based 
mask generator, combines them, and saves the resulting data for downstream analysis.

Key Steps:
1. Captures two images by moving motors: a reference and an initial image.
2. Applies image normalization to emphasize meaningful differences between the two.
3. Uses a segmentation model (e.g., SAM) to generate masks on the normalized image.
4. Combines all masks into a single binary mask for visualization.
5. Stores image dimensions, raw images, masks, and intermediate results in a shared config object.
6. Saves all relevant image outputs (initial, reference, normalized, combined mask) to disk.
"""

import os
import cv2
import numpy as np
from fastmcp import FastMCP

from tools import config
from tools.utils import move_motors_normalize, normalization, combine_masks

mcp = FastMCP("Run First Image")

def run_first_image_core():
    """Captures and processes images for beamline alignment, generating masks and saving data for further steps.""" 
    width, height, im_1, im_0 = move_motors_normalize() # Initial + reference images

    # Normalize image
    image_norm = normalization(config.answer_normalization, im_1, im_0)

    # Creates segmented masks and combines them into one
    mask_image_norm = config.mask_generator.generate(image_norm)
    display_mask = combine_masks(mask_image_norm)
    display_mask = (display_mask > 0).astype(np.uint8) * 255

    # Save data
    config.image_dimensions['width'] = width
    config.image_dimensions['height'] = height
    config.all_image_masks.append(mask_image_norm)
    config.im_0 = im_0
    config.im_1 = im_1
    config.image_norm = image_norm
    config.display_mask = display_mask

    return 'First image has been run'

@mcp.tool
def run_first_image() -> str:
    return run_first_image_core()
