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
path_to_sam = '/home/mgrow/mcp_project/sam_vit_h_4b8939.pth'

# Static parameters - should be specific to setup eventually
angle_rotation = 5
time_exposure = 2
answer_normalization = 0 
pixel_size = 1.172
file_type = 'TIFF1'
camera_type = 'cam1'
cam_name = '1idPG1'
froot = 'pin_alignment'
test_images = sorted(glob.glob(os.path.join(pname, '*.tiff')))


# Dynamic parameters
capture_index = 0
all_image_masks = []
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


# Motor setup - test
# mtr_samX  = DummyMotor('1ide1:m34') # sample x-direction
# mtr_samZ  = DummyMotor('1ide1:m36') # sample z-direction
# mtr_vertRot = DummyMotor('1ide:m9') # vertical rotation axis
# mtr_transRotAx  = DummyMotor('1ide1:m101') # translation of axis under rotation stage 

# Motor setup - actual
mtr_samX = PyEpics.Motor('1ide1:m34') # sample x-direction
mtr_samZ = PyEpics.Motor('1ide1:m36') # sample z-direction
mtr_vertRot = PyEpics.Motor('1ide:m9') # vertical rotation axis
mtr_transRotAx = PyEpics.Motor('1ide1:m101') # translation of axis under rotation stage 
PyEpics.caput(cam_name + ':' + file_type + ':FileName', froot, wait=True)

# Set up SAM model
sam = sam_model_registry["vit_h"](checkpoint=path_to_sam)
sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
mask_generator = SamAutomaticMaskGenerator(sam)