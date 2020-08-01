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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

from tensorflow import keras
from keras.models import load_model

'''
Rolling Window Data Structure
'''
# Note numb joints here means both x,y values, (eg: if BODY_25 we have 25*2 numb joints)
numbJoints   = 98
window_Width = 75

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
'''
dictOfSigns = {
    0: 'help',
    1: 'pain'
}
'''
dictOfSigns = {
    0:"ambulance", 
    1:"help", 
    2:"hospital", 
    3:"pain"
}
# Reference object for LSTM Model
lstm     = None

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

def loadModel():
    global lstm
    try:
        lstm = keras.models.load_model('saved_model.h5', compile=False)
        lstm.summary()
    except Exception as e:
        print("Error In Loading Model", e)
        raise e

def removeConfidenceLevels(data, limit, outputlist):
    xycounter = 0
    for i in range(0,limit):
        if xycounter < 2:
            outputlist.append(np.float(data[i]))
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

    # Reshape for model to read
    reshaped_keypoints = r.getPoints().reshape((1, window_Width, numbJoints))
    
    # Load Keras Model
    global lstm
    try:
        predictions = lstm.predict([reshaped_keypoints])
        #print(predictions)
        guess = np.argmax(predictions)
        word = dictOfSigns[guess] + "-" + str(round(float(np.max(predictions)),2))
    except Exception as e:
        print("Error in prediction", e)
    
    return word

