# Using APS Alignment Servers

## Setup
Required for this setup, a SAM model checkpoint must be used. Download the vit_h version from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file and place it inside the mcp_project folder.


## Cline
Cline is a Visual Studio Code extension that plays nicely with custom MCP servers. Install Cline from the extensions store and navigate to the 'MCP Servers' page. Under 'Installed' there should be a button called 'Configure MCP Servers'. When clicking on this, it will launch a .json configuration file. 
You can enter in the server information here as shown, providing information for your local setup and API key:
```
{
  "mcpServers": {
    "align_sample": {
      "disabled": false,
      "timeout": 120,
      "type": "stdio",
      "command": "/home/mgrow/venv/bin/python",
      "args": [
        "/home/mgrow/mcp_project/main_server.py"
      ],
      "env": {
        "API_KEY":  "sk-proj-psp..."
      }
    }
  }
}
```
Reload Cline and you should see that the server has been activated in your installed servers area.

### Connecting an LLM

Cline uses a variety of different models for running servers.

To use a general LLM I recommend using OpenRouter as the API Provider and deepseek/deepseek-chat:free as the model if you're looking for a free version with decent usage. You will have to generate your own OpenRouter API Key on their site.

To use Argo, Argonne's internal setup, use the linked argo_bridge repository. (https://github.com/AdvancedPhotonSource/argo_bridge/tree/main) To run Cline using Argo, follow the downstream_config.md on their repo. You need to run argo_bridge.py while typing prompts into Cline.

## Understanding the Architecture
This is formatted to have one server, *sequential_server.py*, also called the "Align Sample" server, that holds five tools. Four tools are the four steps of the alignment workflow and the last tool is the full workflow which calls the full alignment process in one easily-callable step. 

1. Run first image: Captures and processes images for beamline alignment, generates masks and saves data for further steps
2. Get coordinates: Automatically identifies and returns beam and object coordinates from an input image and saves horizontal beam edges.
3. Run all images: Processes a series of images to analyze and track midpoint positions through rotation, using segmentation masks and motor control simulation, then fits and visualizes the resulting data.
4. Center pin: Automates pin alignment and verification using motor control, image processing, and segmentation.

All functions used more than once can be found in *utils.py* in the tools folder. This allows for easier maintenance of code and reduces repetition. All parameters can be found in *config.py*. By reading and overwriting parameters through this centralized file, we ensure consistency across modules without needing to pass excessive parameters throughout the code.

## Adapting at APS
After ssh-ing into an APS-hosted machine, set up the motors and camera to match those being used at the beamline and run the program in the same way. Testing is in progress and further motor calibration will most likely be needed.

## Running the Server
This server can be run by typing 'align sample' into the Cline terminal. 
This will run on Cline and produce a confirmation message when the process is finished. 
