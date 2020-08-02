#/usr/bin/env/python
# to start window to record video;
# created by matthew; nebulaM78 team; capstone 2020;
# Loading Model and running tests

# Load Open Pose to process input video
import sys
import cv2
import os, re, os.path
from sys import platform
import argparse
import errno
import shutil
import numpy as np
import json
import glob
import time
import video_tools as REC
import removeConfidenceAndAppend as RCA
import tempfile
import RollingWindow as RW
# needed to save non-string python object; eg dictionary;
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle


PREFIX =  "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\"
json_path =  "C:\\Users\\yongw4\\Desktop\\JSON\\"
op_videopath = PREFIX +  "result.avi"
# where to save your recording?
raw_videopath = PREFIX + 'hello_world.avi'

def offline_predict(PREFIX, write_vname, write_OP_name, json_path, signtime = 3, saved_model = 'saved_model.h5'):
	'''
	args:
		- PREFIX; the directory to save:
				- write_vname; the filename of the self recorded video;
				- write_OP_name; the filenmae for the openpose-processed video;
		- json_path; to store all the json files;
		- signtime; how long do you want to capture the actual sign;
		- saved_model; where is our pretrained lstm model?
	return:
		- a list of the predictions at each 15-frame
	function:
		- offline prediction;
		- sanity check on the model;
	'''

	#------------------------------------------------------
	# setting up the reference paths;
	#------------------------------------------------------
	
	# save the openpose-processed video as well;
	op_videopath = PREFIX +  write_OP_name
	# where to save your recording?
	raw_videopath = PREFIX + write_vname

	# first, check for video directory
	# Try to create these directories, and throw error if cant
	print("checking the directory for the videos")
	try:
		os.makedirs(PREFIX)
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise
		# Ignore error if it is a "directory exists" error
		pass
	except Exception as e:
		print('OOPS', e)

	#------------------------------------------------------
	# create a video;
	#------------------------------------------------------
	# start now; note; signtime is the time interval where you perform the sign;
	REC.record_video(raw_videopath,  signtime)

	#------------------------------------------------------
	# setting up openpose and its parameters;
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
		# setting up json;
		#------------------------------------------------------
		# JSON save path
		if os.path.exists(json_path):
			print('Removing result_json to clear for a new video')
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
			print('OOPS', e)
	
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

	#-----------------------------------------------------------------------
	# PREDICTING THE FUTURE!!!
	# make the prediction at every stepsize: 15 frames;
	# at each stepsize, we will grab 75-frame and pass it to the rolling window;
	# ... then to the model for the prediction;
	#-----------------------------------------------------------------------
	# Tensorflow, Keras imports
	import tensorflow as tf
	from tensorflow import keras
	from keras.models import load_model
	from keras.layers import Dense, Dropout, LSTM
	#saved_model = 'saved_model.h5'
	model = keras.models.load_model(saved_model, compile=False)
	model.summary()

	# hardcode for now;
	dictOfSigns = {0:"ambulance", 1:"help", 2:"hospital", 3:"pain"}
	window_Width = 75
	numbJoints = 98

	# initialize a rolling window;
	matt_window = RW.RollingWindow(window_Width,numbJoints)
	
	# get a sorted list of all the json files of the recorded videos;
	LIST = sorted(glob.glob(json_path + "*.json"))

	# store the (prediction + prob) at each stepsize;
	FUTURE_LIST = []

	# pass in one chunk of 75 frames at a time;
	n = 0
	stepsize = 15
	STOP_FLAG = False
	init_len = len(LIST)
	while(STOP_FLAG != True):
		start_list = stepsize*n
		end_list = (stepsize*n) + window_Width
		# if it goes past the end of the List, 
		# then "extend" by filling up its duplicate (of the last element);
		if (end_list >= init_len):
			diff = abs(init_len - end_list)
			duplicate_elem = LIST[init_len - 1]
			for i in range(0, diff):
				 LIST.append(duplicate_elem)
			# at this stage, we have processed each frame of the video;
			if(start_list > (init_len - 1)):
				STOP_FLAG = True
		# now, extract the 75-size chunk at step = 15;
		# this could be viewed as a sliding window;
		chunk = LIST[start_list:end_list]
		# start at a clean slate; to prevent overlapping;
		matt_window.clearWindow()
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
		index = np.argmax(predictions[0])
		prob = np.max(predictions[0])
		predicted = dictOfSigns[index]
		print('sliding window: ', n, '-- Guessed Sign:', predicted, '; probability:', prob, '\n')

		# storing the information;
		FUTURE_LIST.append((predicted, prob))	
		# increment the step size to check next step 75-frame
		n +=1

	#-----------------------------------------------------------------------
	# sanity checks
	# 1. check whether openpose processed frames and video's fps coincide
	#   1.1. sometimes camera's fps fluctuates;
	#   1.2. to check whether i have implemented the sliding window correctly;
	# 2. print out the list of predictions with its corresponding probability;
	#   2.2 compare against the saved openpose-processed video;
	#-----------------------------------------------------------------------
	
	print('sanity check - 01')
	# hardcode this for now;
	CAMERA_FPS = 30
	total_frames = CAMERA_FPS*signtime
	num_slide = total_frames/stepsize
	print("the sliding window should have been slid: ", num_slide)
	print("the sliding window had been slid: ", n-2)

	print('\n sanity check - 02')
	print('all the predictions and its corresponding probability: \n ')
	print(FUTURE_LIST)
	return FUTURE_LIST

# test driver;
if __name__ == '__main__':
	#------------------------------------------------------------------------
	# CHANGE TO YOUR OWN REFERENCE PATH
	# make sure the videos and the JSON do not share the same directory!!
	# JSON directory will be purged every time to start at a clean slate;
	# explanation:
	#   PREFIX - the directory to store:
	#           - "write_OP_name": openpose_processed video
	#           - "write_vname": self-recorded video
	# json_path: to store all the json's; see the remark above;
	#------------------------------------------------------------------------
	
	PREFIX =  "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\"
	json_path =  "C:\\Users\\yongw4\\Desktop\\JSON\\"
	write_OP_name =  "result.avi"
	write_vname = 'hello_world.avi'
	
	# how long do you want to capture the actual sign;
	signtime = 9
	# import the pretrained model;
	saved_model = 'saved_model.h5'

	FUTURE_LIST = offline_predict(PREFIX, write_vname, write_OP_name, json_path, signtime, saved_model)
	
	# a wrap around;
	# [ISSUE] could not stream the saved video in the same script!!! 
	# so, run the video-script part in another place;
	# to do so, save the (list) predictions first;
	list_path = PREFIX + 'future_prediction.txt'

	with open(list_path, 'wb') as fp:
		pickle.dump(FUTURE_LIST, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("the predictions have been saved as: ", list_path)
