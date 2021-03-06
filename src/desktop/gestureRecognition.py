# created by matthew; nebula-m78 team;

import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pprint as pp

MODEL_35 = True
modelpath = None
numbJoints   = 98
if MODEL_35:
	# Note numb joints here means both x,y values, (eg: if BODY_25 we have 25*2 numb joints)
	window_Width = 35
	modelpath = "../training/training-files/frame-35/iter-01/fmodel.h5"
else:
	modelpath = "../training/training-files/frame-75/iter-01/fmodel.h5"
	window_Width = 75

############### INITIALIZATIONS ##################

#### Tensorflow Imports ####

# Here, we can import tensorflow + keras + Machine Learning Libraries to load models
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

'''
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
'''

memLimit = 256 #128, 256, 512, 1024 
if gpus:
	# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
	try:
		tf.config.experimental.set_virtual_device_configuration(
			gpus[0],
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16)])
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Virtual devices must be set before GPUs have been initialized
		print("Error occured! {}".format(e))
	except Exception as e:
		print(str(e))
else:
	print("no gpus?")


from tensorflow import keras
from tensorflow.keras.models import load_model

'''
Rolling Window Data Structure
'''
import RollingWindow as RW

# Instantiate the rolling window for use later
print("Creating Rolling Window")
r = RW.RollingWindow(window_Width,numbJoints)
print("Finished Created Rolling Window, Window Width = {} & NumbJoints = {}".format(window_Width, numbJoints))

########## KERAS IMPORT ############

# Signs that define output
dictOfSigns = {
	0:"ambulance",
	1:"help", 
	2:"pain", 
	3:"hospital", 
	4:"pain"
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
		lstm = keras.models.load_model(modelpath, compile=False)
		lstm.summary()
	except Exception as e:
		print("Error In Loading Model", e)
		raise e

def offset_translation(array, reference):
	'''
		purpose: for translational invariance;
		arg:
			array; must be in the  shape of (1, number_of_joints, 3);
			refrence: the refrence to be subtracted from; presumable the shoulder;
	'''
	broadcast = np.array([reference, 0, 0], dtype=np.float32)
	shifted = array-broadcast
	return shifted

def removeConfidenceNumpy(datum):
	'''
	Takes in datum object and get pose, and both hand keypoints
	'''
	
	# translational invariant;
	shoulder_center = datum.poseKeypoints[0][3][0]
	#print("shoulder_center:", shoulder_center)
	body_keypoints = offset_translation(datum.poseKeypoints, shoulder_center)
	lefthand_keypoints = offset_translation(datum.handKeypoints[0], shoulder_center)
	righthand_keypoints = offset_translation(datum.handKeypoints[1], shoulder_center)
	
	# We want to keep only first two columns for all keypoints (ignore confidence levels)
	posePoints = body_keypoints[0][1:8,0:2]  # Slice t  o only take keypoints 1-7 removing confidence
	lefthand   = lefthand_keypoints[0][:,0:2] # Get all hand points removing confidence
	righthand  = righthand_keypoints[0][:,0:2] # Get all hand points removing confidence
	
	# Concatenate all numpy matrices and flatten for rolling window storage (as a row)
	keypoints = np.vstack([posePoints,lefthand,righthand]).flatten()
	
	# Return this as the keypoint to be added to rolling window
	return keypoints

############### Main Translation Function ####################

# Translation Module
def translate(datum):
	'''
	Converting Input Keypoints as numpy array in Yick's GitHub dataset format
	'''
	# Output String Variable of Translated word (sentence in future)
	word = "Init"
	try:
		test = len(datum.poseKeypoints[0])
	except Exception as e:
		# Test will return an error if no one is seen as index[0] does not exists
		# Notify user that no one is seen
		word = "No hands!"
		return word, 0 

	# Check if hands are detected
	lefthandempty  = np.count_nonzero(datum.handKeypoints[0] == -1)
	righthandempty = np.count_nonzero(datum.handKeypoints[1] == -1)
	pp.pprint(datum.handKeypoints[0][:,0:2])
	#pp.pprint(datum.handKeypoints[1])
	print(lefthandempty, righthandempty)
	if (lefthandempty > 21) or (righthandempty > 21): 
		return "No hands!", -100

	# Continue to process if we can detect  
	kp = removeConfidenceNumpy(datum)
	# Add to rolling window
	if r.addPoint(kp) == False:
		# Unable to append to keypoints as issue with data shape
		return 'No hands!', 0
	# Reshape for model to read
	reshaped_keypoints = r.getPoints().reshape((1, window_Width, numbJoints))
	# Load Keras Model
	global lstm 
	try:
		predictions = lstm.predict([reshaped_keypoints])
		'''
		if abs( (np.max(predictions) - np.min(predictions)) ) < 0.9 :
			r.resetWindow()
		'''
		guess = np.argmax(predictions)
		word = dictOfSigns[guess] + "-" + str(round(float(np.max(predictions)),2))
	except Exception as e:
		print("Error in prediction", e)
		word = 'Error'
	return word, np.max(predictions)

# test driver;
if __name__ == '__main__':
	test = np.zeros([1,21,3])
	reference = 1000
	array = offset_translation(test, reference)
	print("test: ", test)
	print("array: ", array)
