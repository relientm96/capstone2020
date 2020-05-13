# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import base64
import re
import numpy as np

# Own Library
import gestureRecognition as gr

# Hold parameters for OpenPose as a dictionary
params = dict()
###### SET OPEN POSE FLAGS HERE ############
# Custom Params 
# (refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
# for more parameters)
params["net_resolution"] = "320x320"
params["hand"] = True
params["hand_net_resolution"] = "328x328"

############################################

# Importing OpenPose 
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/Release');
            os.environ['PATH']  = os.environ['PATH']  + ';' + dir_path + "/bin" 
            import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Add others in path?
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

    # Initialize Gesture Recognition Program + OPENPOSE Wrapper
    gr.initOpenPoseLoad()
    print("Gesture Recognition System Started!")
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose Wrapper Started!")

except Exception as e:
    print(e)
    sys.exit(-1)

def processFrames(inputImageUri):
    '''
    Receives input image url from browser as a base64 string
    Decodes as numpy image array and processes using openpose
    Returns encoded base64 url string for processed openpose image
    '''
    try:
        b64_string = inputImageUri.split(',')[0]
        b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
        encoded_string = base64.b64decode(b64_string)
        # Send image to OpenPose for processing
        jpg_as_np = np.frombuffer(encoded_string, dtype=np.uint8)
        frame = cv2.imdecode(jpg_as_np, flags=1)

        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Pass in datum object to send keypoints to gesture recognition module
        word = "Word: " + gr.translate(datum)

        # Adding all of these into image
        image = datum.cvOutputData
        font = cv2.FONT_HERSHEY_SIMPLEX 
        org = (50, 50) 
        fontScale = 1
        color = (255, 255, 0) # Color chosen from BGR values (0-255) 
        thickness = 2
        image = cv2.putText(image, word, org, font,  
                        fontScale, color, thickness, cv2.LINE_AA)

        # Encode Image in base64
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        #print("Processed image in base64",jpg_as_text)
        # Process Image here
        return jpg_as_text

    except Exception as e:
        print(e)
        sys.exit(-1)
        return "noimage"

        
