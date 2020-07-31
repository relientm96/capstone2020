
# Loading Model and running tests

# Load Open Pose to process input video
import sys
import cv2
import os
from sys import platform
import argparse
import errno
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('video_path', help='Recorded Video Path, eg: python run.py <vid_path>')
args = parser.parse_args()

print(args.video_path)

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/openpose-python/Release')
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/openpose-python/Release;' +  dir_path + '/openpose-python/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python')
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params
    params = dict()

    # Static parameters

    # Turn off display for faster rendering
    params["display"]             = 0   
    params["model_folder"]        = "openpose-python/models/"
    params["hand"]                = True
    ## Configurations for openpose (subject to change) ##
    # Pose net resolution
    params["net_resolution"]      = "336x336"
    # Hand Net Resolution
    params["hand_net_resolution"] = "328x328"
    # Type of pose model
    params["model_pose"]          = "BODY_25"
    # Indexed 0, so we save 64 frames
    #params["frame_first"]          = 10
    # Input video path 
    params["video"] = args.video_path

    # JSON save path
    json_path = os.path.join("result_json", "")

    if os.path.exists(json_path):
        print('Removing result_json to clear for new video')
        shutil.rmtree(json_path)

    # Try to create these directories, and throw error if cant
    try:
        os.makedirs(json_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        # Ignore error if it is a "directory exists" error
        pass
    except Exception as e:
        print(e)

    # Check that path creation is correct
    print('Writing JSON to:  ', json_path)
    # Set them to save json + open pose output video to the following paths
    params["write_json"]  = json_path

    # Render video just incase for sanity check
    vid_path  = os.path.join("result.avi")
    params["write_video"] = vid_path
    print('Writing Video to: ', vid_path) 

    # Starting OpenPose
    try:
        opWrapper = op.WrapperPython(3)
        opWrapper.configure(params)
        opWrapper.execute()
    except Exception as e:
        print(e)
        sys.exit(-1)

except Exception as e:
    print(e)
    sys.exit(-1)

####################################################################################################
# Processing on this video's data in a way that the model understands

import json

# Helper function to remove confidence levels and append to keypoint list
def removeConfidenceAndAppend(data, limit, outputlist):
    xycounter = 0
    for i in range(0,limit):
        if xycounter < 2:
            outputlist.append(str(data[i]))
            xycounter += 1
        else:
            xycounter = 0
    return outputlist

# Number of keypoints
numb_keypoints = 98
# Number of frames per gesture reformat
frames_length = 70
extracted_kp = []

n = 0
for j in os.listdir(json_path):
    filepath = os.path.join(json_path,j)
    if ( n > frames_length-1 ):
        break
    else:
        with open(filepath) as jsonfile:
            keypoints = json.load(jsonfile)
            kp = []
            # Only load the first eight points for pose
            removeConfidenceAndAppend(keypoints['people'][0]['pose_keypoints_2d'], 24, kp)
            # Load all hand points
            removeConfidenceAndAppend(keypoints['people'][0]['hand_left_keypoints_2d'],  61, kp)
            removeConfidenceAndAppend(keypoints['people'][0]['hand_right_keypoints_2d'], 61, kp)              
            extracted_kp.append(kp)
    n += 1

import numpy as np

# Convert to numpy array
extracted_kp = np.array(extracted_kp)
print('Input Keypoints Shape', extracted_kp.shape)
print('N result = ', n)

if (extracted_kp.shape[0] < frames_length):
    # Zero pad if < 70 frames
    print('Less than', frames_length ,'frames detected!')
    print('Difference: ', frames_length - len(extracted_kp) )
    for i in range(len(extracted_kp), frames_length):
        extracted_kp = np.vstack( (extracted_kp, np.zeros(numb_keypoints)) )
    print('Now shape is:', extracted_kp.shape)

print(extracted_kp)
####################################################################################################

# Tensorflow, Keras imports
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM

model = keras.models.load_model('first_lstm.h5', compile=False)
model.summary()

dictOfSigns = {
    'help': 0,
    'pain': 1
}

# Convert keypoints to np array and reshape
extracted_kp = extracted_kp.reshape((1, frames_length, numb_keypoints))
print('New Shape:', extracted_kp.shape)

predictions = model.predict([extracted_kp])
print("All Prediction Probabilities:")
print(predictions)
print('----- Result ----- ')
for i in predictions:
    guess = np.argmax(i)
    for key,value in dictOfSigns.items():
        if value == guess:
            print('Input video:', args.video_path)
            print('Guessed Sign:', key, 'with probability:', np.max(i))