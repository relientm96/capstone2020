import sys
import cv2
import os
from sys import platform
import argparse
import time

#### KERAS IMPORTS ####
#import tensorflow as tf

def initOpenPoseLoad():
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
            print("Successfully Loading Gesture Recognition Program")
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
    except Exception as e:
        print(e)
        sys.exit(-1)

######################## Some Helper Functions #########################
# @params : datum = OpenPose Datum Object to access keypoints
# @return : 2D array of OpenPose Body outputs 
# (refer https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)
def getBodyKeyPoints(datum):
    return datum.poseKeyPoints

# Output Keypoints of Hand from OpenPose
# @params : datum = OpenPose Datum Object to access keypoints
# @params : whichHand = 0 for LeftHand, 1 for RightHand 
def handKeyPoints(datum, whichHand):
    if (whichHand > 1) and ( whichHand < 0):
        print("Wrong Argument Value on Choosing Which Hand!")
        return None
    else:
        # Return chosen hand's keypoints
        return datum.handKeypoints[whichHand]

# Print all keypoints
# @params : datum = OpenPose Datum Object to access keypoints
def printKeyPoints(datum):
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
########################################################################

# Translation Module
def translate(datum):
    # Output String Variable of Translated word (sentence in future)
    word = "Hold Translated Word"
    #############################
    # PERFORM TRANSLATION HERE  #
    # --> Input your datum keypoints to your model 
    # --> Get output prediction from model
    # --> Set your output under the "word" string variable
    
    # Run test program to check
    printKeyPoints(datum)
    # Use Nose Coordinates to change word (for now)
    noseCoordinatesX,noseCoordinatesY,noseConfidence  = datum.poseKeypoints[0][0]

    if ( noseCoordinatesX < 250 ):
        word = "Capstone"
    else:
        word = "2020"

    ############################
    return word
