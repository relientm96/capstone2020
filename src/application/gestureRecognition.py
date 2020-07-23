import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pprint as pp

#### Tensorflow Imports ####
# Here, we can import tensorflow + keras + Machine Learning Libraries to load models
import tensorflow as tf
from tensorflow import keras

def initOpenPoseLoad():
    '''
    Import OpenPose Library to use datum API
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

'''
Rolling Window Data Structure
'''

# Number of Joints

# Note numb joints here means both x,y values, (eg: if BODY_25 we have 25*2 numb joints)
numbJoints = 50
window_Width = 5

class RollingWindow:
    def __init__(self, window_Width, numbJoints):
        self.window_Width   = window_Width
        self.numbJoints     = numbJoints
        # Create a 2D Matrix with dimensions
        # window_Width X number of joints
        self.points         = np.zeros(shape=(window_Width,numbJoints))
    
    def getPoints(self):
        # Return the 2D matrix of points
        return self.points
    
    def getWindow_Width(self):
        return self.window_Width

    def getNumbJoints(self):
        return self.numbJoints

    def addPoint(self, arr):
        # Check for same number of joints in input array before we insert
        if ( len(arr) != self.numbJoints ):
            print("Error! Number of items not equal to numbJoints = ", self.numbJoints)
            return
        # Pop out last row in points first
        self.points = np.delete(self.points, self.window_Width-1, 0)
        # Now insert this row to the front of points
        self.points = np.vstack([arr, self.points])

    def printPoints(self):
        pp.pprint(self.points)

# Instantiate the rolling window for use later
print("Creating Rolling Window")
r = RollingWindow(window_Width,numbJoints)
print("Finished Created Rolling Window, Window Width = {} & NumbJoints = {}".format(window_Width, numbJoints))

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

    '''
    Converting Input Keypoints as numpy array in Yick's GitHub dataset format
    '''
    # We use 0 index for the first person in frame, ignore the others
    keypoints = datum.poseKeypoints[0]
    if len(keypoints) < numbJoints/2:
        print('No full number of joints, skip this')
        return 'Not Yet Initializzed'
    
    # Flatten puts the whole 2D matrix as one row vector
    kp = keypoints.flatten()
    # Delete every 3rd element, the confidence level
    kp = np.delete(kp, np.arange(2, kp.size, 3))

    # Add to rolling window
    r.addPoint(kp)
    print(r.getPoints())
    print('\n----------------------------------------------------------------------------\n')

    noseCoordinatesX = datum.poseKeypoints[0][0][0]
    if ( noseCoordinatesX < 250 ):
        word = "Capstone"
    else:
        word = "2020"
    ############################
    return word

