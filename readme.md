# Separated Beamline Alignment with Motor Code

These files encompass the code integrated into the Dash application that was previously used to align the beamline and samples during the setup process of APS experiments. 

The previous code can be found here: https://github.com/AdvancedPhotonSource/auto_beamline_alignment_tomo 

We extracted the code from the Dash interface for later implementation with a large language model. This code is then wrapped into one or multiple servers that can be integrated using model context protocol (MCP).

## Python Files

### Setup
For either setup, a SAM model checkpoint must be used. Download the vit_h version from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file 

To run this code, first the user must decide whether they are running it with test images or using the APS hardware. 

#### Setup with Test Images
To run with test images, download the image files from this zip file and place in a folder in your environment.
https://anl.box.com/s/bsur0f5xc5grrsq7wjs3jbiepyzxboh3
Ensure that the proper path is set in the *pname* parameter in all files. 

Additionally, ensure that in *move_motor()* the code for actual image acquisition is commented out and the code for test image acquisition is not commented out. Also, check that the section for setting up motors has the actual set up commented out and the test setup using the *DummyMotor* class is uncommented.

#### OR Setup with APS Hardware
To run in the APS, ensure that in *move_motor()* the code for actual image acquisition is not commented out and the code for test image acquisition is commented out. Also, check that the section for setting up motors has the test setup using the *DummyMotor* class commented out and actual set up is uncommented.

### Running the Program
This code follows in a series of 4 files. After performing one of the setup steps above, run the code in this order.

#### Step 1: run_first_image.py
This code performs the first step of beamline alignment by moving motors, capturing and optionally normalizing images, generating segmentation masks using the SAM model, and displaying interactive plots for user coordinate selection.

This code will launch three plots in the browser. On any plot, pick an (x,y) coordinate that in inside the beam but outside of the object, and another (x,y) coordinate that is inside the object and enter them into the terminal. These will be used for later steps. 

#### Step 2: submit_beam.py
This program loads image masks and user-selected coordinates, identifies the horizontal edges of a beam within the mask at those coordinates by calculating the midpoint and edge positions, then saves these edge values for subsequent processing.

#### Step 3: run_all_images.py
This code performs automated beamline alignment by moving motors, capturing and normalizing images, generating segmentation masks with SAM, analyzing edge positions, fitting a sinusoidal model to midpoint data across rotations, and visualizing the results.

#### Step 4: center_pin_verify.py
This code performs automated beamline alignment by moving motors, capturing and normalizing images, generating segmentation masks with SAM, analyzing object edges, and fitting motor rotation data to optimize positioning.

## Model Context Protocol Servers
The above code is then wrapped into servers that follow the model context protocol (MCP) framework to integrate with a large language model. 

Server 1: Run First Image - performs the functionality of run_first_image.py described above

Server 2: Get Coordinates - this finds two (x,y) coordinates on the object and beam using the normalized image

Server 3: Submit Beam - performs the functionality of submit_beam.py described above

Server 4: Run All Images - performs the functionality of run_all_images.py described above

Server 5: Center Pin and Verify - performs the functionality of center_pin_verify.py described above

### Setup
For testing purposes, Cline can be used in Visual Studio Code in order to test the calling of tools and the setup of the servers.

In the *cline_mcp_settings.json* file, add in each server in the following format: 
```
"run_first_image": {
  "disabled": false,
  "timeout": 60,
  "type": "stdio",
  "command": "<path-to-python-interpreter>",
  "args": [
    "<path-to-server-script>"
  ],
  "env": {
    "API_KEY": "<your-api-key>"
  }
}
```
This allows the server to be integrated with Cline. In the Cline terminal, the user can enter commands such as 'Submit beam' and Cline will then call the corresponding tool (in this case the submit_beam tool from server 3).
Cline will then ask for permission to use the correct tool and run the code inside of it. This produces identical outputs (images and plots) to the standard python code. Using ```mcp.run()```in the tool body allows for running with Cline, while using ```print(submit_beam.fn())``` allows the user to run the tool through terminal like normal.

## Summary
This code base presents two options for utilizing the functionality of the *beamline_alignment_with_motor_code.py*, with python files and MCP servers. The goal of creating the servers is to soon integrate with Argo, Argonne's internal AI in order for APS scientists to be able to run experiments autonomously. 
