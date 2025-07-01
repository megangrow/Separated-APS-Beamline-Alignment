import cv2
import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from fastmcp import FastMCP
from skimage.measure import label, regionprops
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

mcp = FastMCP("Get Coordinates (Step 1.5)")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
mask_generator = SamAutomaticMaskGenerator(sam)

image_path = 'Image_Outputs/normalized_no_grid.png'
output = '/home/mgrow/Image_Outputs'

def save_data(beam_point, object_point):
    with open(os.path.join('data', 'user_coordinates.pkl'), 'wb') as f:
        pickle.dump({'coord1': beam_point, 'coord2': object_point}, f)

def get_beam_point(mask, beam_contour):
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
    # Check for segmentation masks
    if not masks:
        return None
    # Find smallest mask = object of note + all coordinates on it
    smallest_mask = sorted(masks, key=lambda m: m['area'])[0]['segmentation']

    # Use erosion to 'cushion' the mask to keep points away from edges
    kernel = np.ones((3, 3), np.uint8)
    cushioned_mask = cv2.erode(smallest_mask.astype(np.uint8), kernel, iterations=10)
    ys, xs = np.where(cushioned_mask)
    
    # Ensures there are pixels on the mask
    if len(xs) == 0:
        return None
    
    # Picks a random coordinate on the mask
    idx = np.random.choice(len(xs))
    return (int(xs[idx]), int(ys[idx]))

# This function saves the beam mask and a image with the plotted coordinates, prints coords
def visualize(beam_mask, beam_pt, object_pt):
    cv2.imwrite(os.path.join(output, "beam_mask.png"), beam_mask)
    binary_rgb = cv2.cvtColor(beam_mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(binary_rgb, beam_pt, radius=10, color=(0, 0, 255), thickness=-1)
    cv2.circle(binary_rgb, object_pt, radius=10, color=(255, 0, 0), thickness=-1)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(binary_rgb, cv2.COLOR_BGR2RGB))
    plt.title("Beam and Object Points")
    plt.axis('off')
    plt.savefig(os.path.join(output, "beam_object_points.png"), dpi=150, bbox_inches="tight")
    print(f"Beam point: {beam_pt}")
    print(f"Object point: {object_pt}")

@mcp.tool
def get_coordinates() -> str:
    """Automatically identifies and returns beam and object coordinates from an input image using SAM segmentation and contour detection."""
    # Use the normalized image input to create a binary image
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find beam contour in binary image and create mask
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    beam_contour = max(contours, key=cv2.contourArea)
    beam_mask = np.zeros_like(binary)

    # Use SAM to detect object mask and get point
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)

    # Get beam and object coordinates
    beam_pt = get_beam_point(beam_mask, beam_contour)
    object_pt = get_object_point(masks)

    # Save and visualize
    save_data(beam_pt, object_pt)
    visualize(beam_mask, beam_pt, object_pt)
    return 'Got beam and object coordinates'

if __name__ == "__main__":
    mcp.run()
    #print(get_coordinates.fn())