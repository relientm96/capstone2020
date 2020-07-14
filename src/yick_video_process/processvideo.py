# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

def initOpenPoseLoad():
    '''
    Import OpenPose Library
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../openpose-python/Release')
            os.environ['PATH']  = os.environ['PATH']  + ';' +  dir_path + "/../openpose-python" + ';' + dir_path + "/../openpose-python/bin" 
            import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    except Exception as e:
        print(e)
        sys.exit(-1)

# Initialize Open Pose using function
initOpenPoseLoad()
# Flag Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

"""
Define your custom flag parameters here (as you would do for the cpp case)
as a python dict instead

(refer to include/openpose/flags.hpp for more parameters)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
"""

params = dict()
# Import Models
params["model_folder"] = "../openpose-python/models/"
# Path to video
params["video"]        = "<ADD_YOUR_VID_PATH>"
# Tell OpenPose where to write OpenPose JSON Output to
params["write_json"] =  "<Add your output directory path"> # Warning! this must exists before you start the program
# Body Net Resolution
params["net_resolution"] = "320x320"
# Enable Hands
params["hand"] = True
# Hand Net Resolution
params["hand_net_resolution"] = "328x328"

# This code here allows you to do similar flag thing as cpp type by appending "--<flag>" when you run this script
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

# Starting OpenPose
try:
    opWrapper = op.WrapperPython(3)
    opWrapper.configure(params)
    print("Writing Videos Now")
    opWrapper.execute()
except Exception as e:
    print("An error occured", e)
    sys.exit(-1)