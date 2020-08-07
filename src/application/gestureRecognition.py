import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pprint as pp

############### INITIALIZATIONS ##################

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
import RollingWindow as RW
# Note numb joints here means both x,y values, (eg: if BODY_25 we have 25*2 numb joints)
numbJoints   = 98
window_Width = 75

# Instantiate the rolling window for use later
print("Creating Rolling Window")
r = RW.RollingWindow(window_Width,numbJoints)
print("Finished Created Rolling Window, Window Width = {} & NumbJoints = {}".format(window_Width, numbJoints))

########## KERAS IMPORT ############

# Signs that define output
dictOfSigns = {
    0:"ambulance", 
    1:"help", 
    2:"hospital", 
    3:"pain"
}
# Reference object for LSTM Model
lstm = None

############### Helper Functions ####################

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

def removeConfidenceNumpy(datum):
    '''
    Takes in datum object and get pose, and both hand keypoints
    '''

    # We want to keep only first two columns for all keypoints (ignore confidence levels)
    posePoints = datum.poseKeypoints[0][1:8,0:2]  # Slice to only take keypoints 1-7 removing confidence
    lefthand   = datum.handKeypoints[0][0][:,0:2] # Get all hand points removing confidence
    righthand  = datum.handKeypoints[1][0][:,0:2] # Get all hand points removing confidence

    # Concatenate all numpy matrices and flatten for rolling window storage (as a row)
    keypoints = np.vstack([posePoints,lefthand,righthand]).flatten()
    
    # Return this as the keypoint to be added to rolling window
    return keypoints

############### Main Translation Function ####################

# Translation Module
def translate(datum):

    # Output String Variable of Translated word (sentence in future)
    word = "Init"

    '''
    Converting Input Keypoints as numpy array in Yick's GitHub dataset format
    '''

    try:
        test = len(datum.poseKeypoints[0])
    except Exception as e:
        # Test will return an error if no one is seen as index[0] does not exists
        # Notify user that no one is seen
        word = "Nobody Here!"
        return word
        
    # Continue to process if we can detect  
    kp = removeConfidenceNumpy(datum)
    
    # Add to rolling window
    if r.addPoint(kp) == False:
        # Unable to append to keypoints as issue with data shape
        return 'Error'

    # Reshape for model to read
    reshaped_keypoints = r.getPoints().reshape((1, window_Width, numbJoints))
    
    # Load Keras Model
    global lstm
    try:
        predictions = lstm.predict([reshaped_keypoints])
        guess = np.argmax(predictions)
        word = dictOfSigns[guess] + "-" + str(round(float(np.max(predictions)),2))
    except Exception as e:
        print("Error in prediction", e)
        word = 'Error'
    
    return word

