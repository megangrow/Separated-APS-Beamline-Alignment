from PIL import Image
import glob
import os
import epics as PyEpics
from motor import DummyMotor

class Params:
    def __init__(self):
        angle_rotation = 5
        time_exposure = 2
        pixel_size = 1.172
        capture_index = 0
        file_type = 'TIFF1'
        camera_type = 'cam1'
        cam_name = '1idPG1'
        pname = '/home/mgrow/APS_Test_Images'
        froot = 'pin_alignment'
        output = '/home/mgrow/Image_Outputs'
        test_images = sorted(glob.glob(os.path.join(pname, '*.tiff')))
        ref_im_path = '/home/mgrow/Image_Outputs/reference_image.tiff'
        im_0 = Image.open(ref_im_path)

        mtr_samXE   = DummyMotor('1ide1:m34')
        mtr_samYE   = DummyMotor('1ide1:m36')
        mtr_samOme  = DummyMotor('1ide:m9')
        mtr_aeroXE  = DummyMotor('1ide1:m101')
        # mtr_samXE = PyEpics.Motor('1ide1:m34')
        # mtr_samYE = PyEpics.Motor('1ide1:m36')
        # mtr_samOme = PyEpics.Motor('1ide:m9')
        # mtr_aeroXE = PyEpics.Motor('1ide1:m101')
        # PyEpics.caput(cam_name + ':' + file_type + ':FileName', froot, wait=True)