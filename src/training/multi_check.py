# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import errno
import tempfile
import glob, os
import ffmpeg
import tensorflow as tf

# from nebulaM78;
import file_tools as ftools
import generate_XY as genxy
import json_video2txt as jv2t
import multiprocessing
import time


def PARAMS(video_path, write_path):
	params = dict()
	params['render_pose']               = 1
	params['display']                   = 2
	params['number_people_max']         = 1
	params["model_pose"]                = "BODY_25"
	params["net_resolution"]         = "336x336"
	params["hand"]                      = True
	params["hand_net_resolution"]       = "328x328"
	params['disable_multi_thread']      = True
	params['process_real_time']         = False
	params["fps_max"]             = -1
	params["frame_first"]         = 0
	params["frame_last"]          = 74 
	params['frame_flip']            = False
	params['keypoint_scale']            = 3

	# Import Models
	params["model_folder"] = "C:\\CAPSTONE\\capstone2020\\src\\openpose-python\\models"
	
	params["video"] = video_path
	
	# Tell OpenPose where to write OpenPose JSON Output to
	#write_path = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg"
	
	# check if the directory exists?;
	if not os.path.isdir(write_path):
		try:
			print("creating a directory for the json files;\n")
			os.mkdir(write_path)
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)
	else:
		print('the directory has been created\n')

	params["write_json"] =  write_path # Warning! this must exists before you start the program
	return params

def MAIN(video_path, write_path):
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

	params = PARAMS(video_path, write_path)
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

		"""
		Define your custom flag parameters here (as you would do for the cpp case)
		as a python dict instead
		(refer to include/openpose/flags.hpp for more parameters)
		https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp
		"""

			# Add others in path?
		for i in range(0, len(args[1])):
			curr_item = args[1][i]
			if i != len(args[1])-1: next_item = args[1][i+1]
			else: next_item = "1"
			if "--" in curr_item and "--" in next_item:
				key = curr_item.replace('-','')
				if key not in params:  params[key] = "1"
			elif "--" in curr_item and "--" not in next_item:
				key = curr_item.replace('-','')
				if key not in params: params[key] = next_item

		# Construct it from system arguments
		# op.init_argv(args[1])
		# oppython = op.OpenpoFILE_FORMATython()

		# Starting OpenPose
		opWrapper = op.WrapperPython(3)
		opWrapper.configure(params)
		opWrapper.execute()

	except Exception as e:
		print(e)
		sys.exit(-1)

if __name__ == '__main__':
	video_path = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_1.mp4"
	video_path2 = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\fullbody.mp4"
	write_path = "C:\\Users\\yongw4\\Desktop\\DUMMY_JSON"
	write_path2 = "C:\\Users\\yongw4\\Desktop\\DUMMY_JSON2"

	start_time = time.time()
	proc = [[video_path, write_path], [video_path2, write_path2]]
	for ls in proc:
		## right here
		p = multiprocessing.Process(target=MAIN, args=(ls[0], ls[1]))
		p.start()
		proc.append(p)
	for pp in proc:
		pp.join()
	print("--- %s seconds ---" % (time.time() - start_time))
	
