# argo_bridge custom base url:http://localhost:7285
# argo-proxy custom base url: http://localhost:55068

from fastmcp import FastMCP
from tools.run_first_image_tool import run_first_image_core
from tools.get_coordinates_tool import get_coordinates_core
from tools.run_all_images_tool import run_all_images_core
from tools.center_pin_tool import center_pin_core

mcp = FastMCP("Align Sample")

@mcp.tool
def align_sample() -> str:
    """Aligns the sample by carrying out a 4-step process"""
    results = []
    results.append(run_first_image_core())
    results.append(get_coordinates_core())
    results.append(run_all_images_core())
    results.append(center_pin_core())
    return "\n".join(results)

if __name__ == "__main__":
    mcp.run()