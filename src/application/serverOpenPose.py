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

# Hold parameters for OpenPose as a dictionary
params = dict()
###### SET OPEN POSE FLAGS HERE ############
# Custom Params 
# (refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
# for more parameters)

params["net_resolution"]       = "160x160"
params["hand"]                 = True
params["hand_net_resolution"]  = "296x296"
params['keypoint_scale']       = 3
#params["disable_multi_thread"] = False
params["number_people_max"]    = 1
############################################

# Importing OpenPose 
try:
    '''
    Import OpenPose Library and wrapper
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Import Models
    params["model_folder"] = "../openpose-python/models/"
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

    # Start openpose wrapper
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose Wrapper Started!")

except Exception as e:
    print(e)
    sys.exit(-1)

def removeConfidenceAndShapeAsNumpy(datum):
    '''
    Takes in datum object and get pose, and both hand keypoints 
    and returns a numpy array for prediction
    '''

    # Test and see if we have anyone in image
    try:
        test   = len(datum.poseKeypoints[0])
    except Exception as e:
        # Test will return an error if no one is seen as index[0] does not exists
        # Notify user that no one is seen
        return np.zeros(1)

    # We want to keep only first two columns for all keypoints (ignore confidence levels)
    posePoints = datum.poseKeypoints[0][1:8,0:2]  # Slice to only take keypoints 1-7 removing confidence
    lefthand   = datum.handKeypoints[0][0][:,0:2] # Get all hand points removing confidence
    righthand  = datum.handKeypoints[1][0][:,0:2] # Get all hand points removing confidence

    # Concatenate all numpy matrices and flatten for rolling window storage (as a row)
    keypoints = np.vstack([posePoints,lefthand,righthand]).flatten()
    
    # Return this as the keypoint to be added to rolling window
    return keypoints

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

        # Adding all of these into image
        image = datum.cvOutputData
        
        # Numpy result of extracted keypoints processed
        keypoints = removeConfidenceAndShapeAsNumpy(datum)

        # Encode Image in base64
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        # Process Image here
        return jpg_as_text, keypoints

    except Exception as e:
        print(e)
        sys.exit(-1)
        return "noimage"

def returnKeypointsFlattened(inputImageUri):
    '''
    Returns keypoints generated for this image (flattened for prediction)
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
        return removeConfidenceAndShapeAsNumpy(datum)
    except Exception as e:
        print(e)
        sys.exit(-1)
        return "noimage"