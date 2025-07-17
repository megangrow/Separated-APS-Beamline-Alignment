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
    config.object_pt, mask = get_object_point(masks)
    print(f"Beam point: {beam_pt}")
    print(f"Object point: {config.object_pt}")
    save_images(beam_mask, config.beam_pt, config.object_pt, mask, config.output)

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
    """Returns a random interior point from the smallest segmentation mask (target object)."""
    # Check for segmentation masks
    if not masks:
        return None
    # Find smallest mask = object of note + all coordinates on it
    smallest_mask = sorted(masks, key=lambda m: m['area'])[0]['segmentation']

    # Use erosion to 'cushion' the mask to keep points away from edges
    kernel = np.ones((3, 3), np.uint8)
    cushioned_mask = cv2.erode(smallest_mask.astype(np.uint8), kernel, iterations=10)
    ys, xs = np.where(cushioned_mask)
    
    # Ensures there are pixels on the mask + picks random coordinate
    if len(xs) == 0:
        return None
    idx = np.random.choice(len(xs))
    return (int(xs[idx]), int(ys[idx])), smallest_mask

def save_images(beam_mask, beam_point, object_point, smallest_mask, out):
    """Saves all image outputs from the config object to disk."""
    cv2.imwrite(os.path.join(out, "beam_mask.png"), beam_mask)
    cv2.imwrite("segmented_mask.png", (smallest_mask * 255).astype(np.uint8))

    # Convert beam mask to color for overlay + save
    color_img = cv2.cvtColor(beam_mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(color_img, beam_point, radius=10, color=(0, 0, 255), thickness=-1)     # Red circle
    cv2.circle(color_img, object_point, radius=10, color=(255, 0, 0), thickness=-1)   # Blue circle
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(os.path.join(out, "beam_object_points.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote images -2")

@mcp.tool
def get_coordinates() -> str:
    return get_coordinates_core()