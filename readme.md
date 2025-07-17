# Using APS Alignment Servers

## Setup
Required for this setup, a SAM model checkpoint must be used. Download the vit_h version from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file 

To run this code, first the user must decide whether they are running it with test images or using the APS hardware. 

### Download Files
The required files for this code are *main_server.py* and a folder 'tools' that includes the other files, both placed in your root.

### Setup with Test Images
To run with test images, download the image files from this zip file and place in a folder in your environment.
https://anl.box.com/s/bsur0f5xc5grrsq7wjs3jbiepyzxboh3
Ensure that the proper path is set in the *pname* parameter in all files. 

Additionally, ensure that in *move_motor()* the code for actual image acquisition is commented out and the code for test image acquisition is not commented out. Also, check that the section for setting up motors has the actual set up commented out and the test setup using the *DummyMotor* class is uncommented.

### OR Setup with APS Hardware
To run in the APS, ensure that in *move_motor()* the code for actual image acquisition is not commented out and the code for test image acquisition is commented out. Also, check that the section for setting up motors has the test setup using the *DummyMotor* class commented out and actual set up is uncommented.

### Setup Cline
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
        "/home/mgrow/main_server.py"
      ],
      "env": {
        "API_KEY":  "sk-proj-psp..."
      }
    }
  }
}
```
Reload Cline and you should see that the server has been activated in your installed servers area.

Cline uses a variety of different models for running servers. I recommend using OpenRouter as the API Provider and deepseek/deepseek-chat:free as the model if you're looking for a free version with decent usage. You will have to generate your own OpenRouter API Key on their site.

## Understanding the Architecture
These servers are formatted to have one server, *main_server.py*, also called the "Align Sample" server, that executes four tools in a particular order. This works through the full alignment process in one easily-callable step. 

1. Run first image: Captures and processes images for beamline alignment, generates masks and saves data for further steps
2. Get coordinates: Automatically identifies and returns beam and object coordinates from an input image and saves horizontal beam edges.
3. Run all images: Processes a series of images to analyze and track midpoint positions through rotation, using segmentation masks and motor control simulation, then fits and visualizes the resulting data.
4. Center pin: Automates pin alignment and verification using motor control, image processing, and segmentation.

All functions used more than once can be found in *utils.py* in the tools folder. This allows for easier maintenance of code and reduces repetition. All parameters can be found in *config.py*. By reading and overwriting parameters through this centralized file, we ensure consistency across modules without needing to pass excessive parameters throughout the code.

## Running the Server
This server can be run by typing 'align sample' or some other variation of this command into the Cline terminal. 
This will run on Cline and produce a confirmation message when the process is finished. 

This will also produce a folder of Image Outputs in your root that you can view to see the code progression.
