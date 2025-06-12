from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP


# Define your input/output schemas
class Step1Input(BaseModel):
    number: int

class Step1Output(BaseModel):
    result: int


# Define your tool function
def step1_tool(input: Step1Input) -> Step1Output:
    print(f"[Tool] Received: {input.number}")
    return Step1Output(result=input.number + 1)

# Start the MCP server
if __name__ == "__main__":
    server = FastMCP()
    server.register_tool(step1_tool)
    server.run_stdio() 