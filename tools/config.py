"""
This module contains all parameters that are used throughout the alignment process
"""
import os
import glob
import torch
import epics as PyEpics
from .motor import DummyMotor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Local-specific parameters (change based on setup)
output = '/home/mgrow/Image_Outputs'
pname = '/home/mgrow/APS_Test_Images'

# Static parameters
angle_rotation = 5
time_exposure = 2
pixel_size = 1.172
file_type = 'TIFF1'
camera_type = 'cam1'
cam_name = '1idPG1'
froot = 'pin_alignment'
test_images = sorted(glob.glob(os.path.join(pname, '*.tiff')))


# Dynamic parameters
capture_index = 0
all_image_masks = []
answer_normalization = 0 # TODO: switch to getting user input for this somehow??
user_coordinates = {}
edges = []
params = []
im_0 = None
im_1 = None
image_norm = None
display_mask = None
image_dimensions = {'width': 0, 'height': 0}
beam_pt = None
object_pt = None
offset = None
radius = None
start_theta = None


# Motor setup -test
mtr_samXE   = DummyMotor('1ide1:m34')
mtr_samYE   = DummyMotor('1ide1:m36')
mtr_samOme  = DummyMotor('1ide:m9')
mtr_aeroXE  = DummyMotor('1ide1:m101')

# Motor setup - actual
# mtr_samXE = PyEpics.Motor('1ide1:m34')
# mtr_samYE = PyEpics.Motor('1ide1:m36')
# mtr_samOme = PyEpics.Motor('1ide:m9')
# mtr_aeroXE = PyEpics.Motor('1ide1:m101')
# PyEpics.caput(cam_name + ':' + file_type + ':FileName', froot, wait=True)

# Set up SAM model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
mask_generator = SamAutomaticMaskGenerator(sam)