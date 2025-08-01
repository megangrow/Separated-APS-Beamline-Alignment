# This server contains the tools to run the full sequential alignment process as well
# as each of the four individual steps.
from fastmcp import FastMCP
from tools.run_first_image_tool import run_first_image_core
from tools.get_coordinates_tool import get_coordinates_core
from tools.run_all_images_tool import run_all_images_core
from tools.center_pin_tool import center_pin_core

def align_sample_core() -> str:
    results = []
    results.append(run_first_image_core())
    results.append(get_coordinates_core())
    results.append(run_all_images_core())
    results.append(center_pin_core())
    return "\n".join(results)

mcp = FastMCP("Align Sample")

@mcp.tool
def run_first_image() -> str:
    """Captures initial and reference images of the sample by controlling motors and camera.
    This step prepares raw image data for downstream processing."""
    return run_first_image_core()

@mcp.tool
def get_coordinates() -> str:
    """Processes images to identify key coordinate points of the sample and beamline edges.
    Uses image contour detection and segmentation to define alignment targets."""
    return get_coordinates_core()

@mcp.tool
def run_all_images() -> str:
    """Collects images across a full rotation sequence, tracks the sample position in each frame, and manages motor adjustments to maintain the sample within the camera’s field of view."""
    return run_all_images_core()

@mcp.tool
def center_pin() -> str:
    """Calculates alignment offsets from tracked sample positions and commands motor moves
    to center the sample precisely along the beamline axis.
    Provides alignment verification through trajectory visualization."""
    return center_pin_core()

@mcp.tool
def align_sample() -> str:
    """Runs the complete beamline alignment workflow by sequentially executing:
    capturing initial images, extracting coordinates, processing rotation images,
    and centering the sample using motor adjustments.
    This comprehensive tool automates the full alignment process from start to finish."""
    return align_sample_core()

if __name__ == "__main__":
    mcp.run()