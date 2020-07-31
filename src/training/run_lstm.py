
# Loading Model and running tests

# Load Open Pose to process input video
import sys
import cv2
import os
from sys import platform
import argparse
import errno
import shutil
import numpy as np
import json
import record_video as REC
import removeConfidenceAndAppend as RCA
import tempfile
import RollingWindow as RW

#------------------------------------------------------
# CHANGE TO YOUR OWN REFERENCE PATH
#------------------------------------------------------
PREFIX = "C:\\Users\\yongw4\\Desktop\\JSON\\"
# outputs from openpose: json and its processed video
json_path = PREFIX
op_videopath = PREFIX +  "result.avi"
# where to save your recording?
raw_videopath = PREFIX + 'hello_world.avi'

#------------------------------------------------------
# create a video;
#------------------------------------------------------
# start now; note; signtime is the time interval where you perform the sign;
REC.record_video(raw_videopath,  signtime = 3)

#------------------------------------------------------
# setting up openpose;
#------------------------------------------------------
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
    params["video"] = raw_videopath

    # Static parameters
    params["display"]             = 2
    params['render_pose']         = 1
    params["model_folder"]        = "openpose-python/models/"
    params["hand"]                = True
    params["net_resolution"]      = "336x336"
    params["hand_net_resolution"] = "328x328"
    params["model_pose"]          = "BODY_25"
    params['keypoint_scale']      = 3
    params['number_people_max']   = 1

    #params['frame_flip']            = True
	params["fps_max"]             = -1
    

    #------------------------------------------------------
    # setting up json and video storages;
    # reminder;
    # json_path = PREFIX
    # op_videopath = PREFIX +  "result.avi"
    #------------------------------------------------------

    # JSON save path
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
    params["write_video"] = op_videopath
    print('Writing Video to: ', op_videopath) 
    
    # Check that path creation is correct
    print('Writing JSON to:  ', json_path)
    # Set them to save json + open pose output video to the following paths
    params["write_json"]  = json_path

    #------------------------------------------------------
    # start openpose;
    #------------------------------------------------------
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

#------------------------------------------------------
# Processing on this video's data in a way that the model understands
#------------------------------------------------------
    
window_Width = 75
numbJoints = 98
matt_window = RW.RollingWindow(window_Width,numbJoints)
matt_window.addPoint(kp)
reshaped_keypoints = matth_window.getPoints().reshape((1, window_Width, numbJoints))
	


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