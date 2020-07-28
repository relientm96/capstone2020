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

'''
Rolling Window Data Structure
'''
# Note numb joints here means both x,y values, (eg: if BODY_25 we have 25*2 numb joints)
numbJoints   = 98
window_Width = 70

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
            return False
        # Pop out last row in points first
        self.points = np.delete(self.points, self.window_Width-1, 0)
        # Now insert this row to the front of points
        self.points = np.vstack([arr, self.points])
        return True

    def printPoints(self):
        pp.pprint(self.points)

# Instantiate the rolling window for use later
print("Creating Rolling Window")
r = RollingWindow(window_Width,numbJoints)
print("Finished Created Rolling Window, Window Width = {} & NumbJoints = {}".format(window_Width, numbJoints))

########## KERAS IMPORT ############

# Signs that define output
dictOfSigns = {
    'help': 0,
    'pain': 1
}

lstm = keras.models.load_model('first_lstm.h5')
lstm.summary()

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

def removeConfidenceLevels(data, limit, outputlist):
    xycounter = 0
    for i in range(0,limit):
        if xycounter < 2:
            outputlist.append(str(data[i]))
            xycounter += 1
        else:
            xycounter = 0
    return outputlist

# Translation Module
def translate(datum):
    # Output String Variable of Translated word (sentence in future)
    word = "Test"

    '''
    Converting Input Keypoints as numpy array in Yick's GitHub dataset format
    '''

    # We use 0 index for the first person in frame, ignore the others
    kp = []
    
    # Flatten as one row vector and remove confidence levels
    pose      = removeConfidenceLevels(datum.poseKeypoints[0].flatten(), 24, kp) 
    # Delete every 3rd element, (remove confidence level)

    # Flatten as one row vector and remove confidence levels
    lefthand  = removeConfidenceLevels(datum.handKeypoints[0][0].flatten(), 61, kp) 

    # Flatten as one row vector and remove confidence levels
    righthand = removeConfidenceLevels(datum.handKeypoints[1][0].flatten(), 61, kp) 

    # Add to rolling window
    if r.addPoint(kp) == False:
        # Unable to append to keypoints as issue with data shape
        return 'Fail'
    
    #print(r.getPoints().shape)
    #print(r.getPoints())

    predictions = lstm.predict()
    guess = np.argmax(i)
    for key,value in dictOfSigns.items():
        if value == guess:
            word = key

    return word

