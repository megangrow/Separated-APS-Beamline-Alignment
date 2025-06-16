from fastmcp import FastMCP
import epics

mcp = FastMCP("Get Coordinates (Step 1.5)")

@mcp.tool
def get_coordinates() -> str:
    """Get Coordinates"""
    return 'Coordinates have been gotten'

if __name__ == "__main__":
    mcp.run()