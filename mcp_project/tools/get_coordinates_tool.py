"""
This script processes a normalized beamline image to identify and extract:
- A point within the beam region.
- A point within the smallest segmented object (assumed to be the target object).
- Horizontal edges of the beam for alignment reference.

Key Steps:
1. Thresholds the normalized image to extract the beam contour.
2. Generates segmentation masks using SAM to identify objects.
3. Calculates beam and object coordinates using custom heuristics.
4. Saves binary masks and overlay images for visual verification.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fastmcp import FastMCP

from tools import config
from tools.utils import get_mid_point

mcp = FastMCP("Get Coordinates")

def get_coordinates_core():
    """Automatically identifies and returns beam and object coordinates from an input image and saves horizontal beam edges."""
    # Use the normalized image input to create a binary image
    gray = cv2.cvtColor(config.image_norm, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find beam contour in binary image and create mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    beam_contour = max(contours, key=cv2.contourArea)
    beam_mask = np.zeros_like(binary)

    # Use SAM to detect object mask and get point
    masks = config.mask_generator.generate(config.image_norm)

    # Get beam and object coordinates
    beam_pt = get_beam_point(beam_mask, beam_contour)
    x, y = beam_pt
    mid, min_x, max_x = get_mid_point(config.all_image_masks[0], y, x)
    
    config.edges.append(min_x)
    config.edges.append(max_x)
    config.beam_pt = beam_pt
    result = get_object_point(masks)
    if result is None:
        config.object_pt = None
        mask = None
    else:
        config.object_pt, mask = result
    print(f"Beam point: {beam_pt}")
    print(f"Object point: {config.object_pt}")

    return 'Got beam and object coordinates'


def get_beam_point(mask, beam_contour):
    """Returns a random point located within the filled beam contour mask."""
    # Draw beam contour on mask and fill white
    cv2.drawContours(mask, [beam_contour], -1, 255, thickness=cv2.FILLED)
    # Find all coordinates with white pixels
    ys, xs = np.where(mask == 255)

    # Randomly select a coordinate inside the contour
    indices = np.random.permutation(len(xs))
    for i in indices:
        x, y = xs[i], ys[i]
        return (int(x), int(y))

def get_object_point(masks):
    if not masks:
        print("No masks detected by SAM.")
        return None
    smallest_mask = sorted(masks, key=lambda m: m['area'])[0]['segmentation']
    print(f"Smallest mask area: {np.sum(smallest_mask)} pixels")
    kernel = np.ones((3, 3), np.uint8)
    #cushioned_mask = cv2.erode(smallest_mask.astype(np.uint8), kernel, iterations=10)
    cushioned_mask = smallest_mask.astype(np.uint8)
    print(f"Pixels after erosion: {np.sum(cushioned_mask)}")
    ys, xs = np.where(cushioned_mask)
    if len(xs) == 0:
        print("No pixels left in eroded mask, cannot find object point.")
        return None
    idx = np.random.choice(len(xs))
    return (int(xs[idx]), int(ys[idx])), smallest_mask

@mcp.tool
def get_coordinates() -> str:
    return get_coordinates_core()