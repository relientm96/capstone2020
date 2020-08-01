
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
import glob
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
REC.record_video(raw_videopath,  signtime = 9)

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
			sys.path.append(dir_path + '/../openpose-python/Release')
			os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../openpose-python/Release;' +  dir_path + '/../openpose-python/bin;'
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
	# Flags
	parser = argparse.ArgumentParser()
	args = parser.parse_known_args()

	# Custom Params
	params = dict()
	params["video"] = raw_videopath

	# Static parameters

	params['display']                   = 0

	params["model_folder"]        = "../openpose-python/models/"
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
	'''
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
	print('HELLO')
	'''
	# Check that path creation is correct
	print('Writing JSON to:  ', json_path)
	# Set them to save json + open pose output video to the following paths
	params["write_json"]  = json_path

	# Render video just incase for sanity check
	params["write_video"] = op_videopath
	print('Writing Video to: ', op_videopath) 
	
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
# Tensorflow, Keras imports
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Dense, Dropout, LSTM

print("HERE?")
model = keras.models.load_model('saved_model.h5', compile=False)
model.summary()

dictOfSigns = {0:"ambulance", 1:"help", 2:"hospital", 3:"pain"}
 
window_Width = 75
numbJoints = 98
matt_window = RW.RollingWindow(window_Width,numbJoints)
	
# get a sorted list of all the json files of the recorded videos;
LIST = sorted(glob.glob(json_path + "*.json"))
# pass in one chunk of 75 frames at a time;

n = 0
stepsize = 15
STOP_FLAG = False
while(STOP_FLAG != True):
	start_list = stepsize*n
	end_list = (stepsize*n) + 75
	# if it goes past the end of the List, 
	# then "extend" by filling up its duplicate (of the last element);
	if (end_list >= len(LIST)):
		diff = abs(len(LIST) - end_list)
		duplicate_elem = LIST[len(LIST)-1]
		for i in range(0, diff):
			 LIST.append(duplicate_elem)
		# at this stage, we have reached the end;
		STOP_FLAG = True
	# now, extract the 75-size chunk at step = 15;
	chunk = LIST[start_list:end_list]
	# fill up the rolling window ;
	for i in range(0, window_Width):
		with open(chunk[i]) as jsonfile:
			keypoints = json.load(jsonfile)
			reformat_chunk = RCA.removeConfidenceAndAppend(keypoints)
			matt_window.addPoint(reformat_chunk)
	# finished filling up the rolling window;
	# ready to pass it to the model;
	reshaped_keypoints = matt_window.getPoints().reshape((1, window_Width, numbJoints))
	# start predicting based on this 75-size chunk;
	predictions = model.predict([reshaped_keypoints])
	print("All Prediction Probabilities:")
	print(predictions)
	print('----- Result ----- ')
	j = 1
	for i in predictions:
		index = np.argmax(i)
		prob = np.max(i)
		predicted = dictOfSigns[index]
		print(j, ': Guessed Sign:', predicted, '; probability:', prob)
		j += 1

	# increment the step size to check next step 75-frame
	n +=1

