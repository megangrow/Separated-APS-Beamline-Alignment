# Separated Beamline Alignment with Motor Code

These files encompass the code integrated into the Dash application that was previously used to align the beamline and samples during the setup process of APS experiments. 

The previous code can be found here: https://github.com/AdvancedPhotonSource/auto_beamline_alignment_tomo 

We extracted the code from the Dash interface for later implementation with a large language model. This code will be wrapped into one or multiple servers that can be integrated using model context protocol (MCP).

## Setup
For either setup, a SAM model checkpoint must be used. Download the vit_h version from https://github.com/facebookresearch/segment-anything?tab=readme-ov-file 

To run this code, first the user must decide whether they are running it with test images or using the APS hardware. 

### Setup with Test Images
To run with test images, download the image files from this zip file and place in a folder in your environment.
https://anl.box.com/s/bsur0f5xc5grrsq7wjs3jbiepyzxboh3
Ensure that the proper path is set in the *pname* parameter in all files. 

Additionally, ensure that in *move_motor()* the code for actual image acquisition is commented out and the code for test image acquisition is not commented out. Also, check that the section for setting up motors has the actual set up commented out and the test setup using the *DummyMotor* class is uncommented.

### OR Setup with APS Hardware
To run in the APS, ensure that in *move_motor()* the code for actual image acquisition is not commented out and the code for test image acquisition is commented out. Also, check that the section for setting up motors has the test setup using the *DummyMotor* class commented out and actual set up is uncommented.

## Running the Program
This code follows in a series of 4 files. After performing one of the setup steps above, run the code in this order.

### Step 1: run_first_image.py
This code performs the first step of beamline alignment by moving motors, capturing and optionally normalizing images, generating segmentation masks using the SAM model, and displaying interactive plots for user coordinate selection.

This code will launch three plots in the browser. On any plot, pick an (x,y) coordinate that in inside the beam but outside of the object, and another (x,y) coordinate that is inside the object and enter them into the terminal. These will be used for later steps. 

### Step 2: submit_beam.py
This program loads image masks and user-selected coordinates, identifies the horizontal edges of a beam within the mask at those coordinates by calculating the midpoint and edge positions, then saves these edge values for subsequent processing.

### Step 3: run_all_images.py
This code performs automated beamline alignment by moving motors, capturing and normalizing images, generating segmentation masks with SAM, analyzing edge positions, fitting a sinusoidal model to midpoint data across rotations, and visualizing the results.

### Step 4: center_pin_verify.py
This code performs automated beamline alignment by moving motors, capturing and normalizing images, generating segmentation masks with SAM, analyzing object edges, and fitting motor rotation data to optimize positioning.

## Summary
This program creates outputs for the user to review.

- The original image
- The normalized image
- The segmented image (multiple)
- A plot of the midpoint positions vs rotation angles
