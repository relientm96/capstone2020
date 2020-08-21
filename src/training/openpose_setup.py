# From Python
# It requires OpenCV installed for Python
import sys
import cv2
from sys import platform
import argparse
import errno
import tempfile
import glob, os
import ffmpeg
import tensorflow as tf

# from nebulaM78;
import time
import multiprocessing as mp
import shutil
import numpy as np
import moviepy.editor as mp
import moviepy.video.fx.all as vfx

import file_tools as ftools
import generate_XY as genxy
import json_video2txt as jv2t
import video_tools as VID
import synthetic_tools as SYNTH


def PARAMS():
	params = dict()
	# preamble and computation;
	params["model_folder"]              = "C:\\CAPSTONE\\capstone2020\\src\\openpose-python\\models"
	params["model_pose"]                = "BODY_25"
	params['render_pose']               = 0
	params['display']                   = 0
	params['number_people_max']         = 1
	params['disable_multi_thread']      = True
	params['process_real_time']         = False
	params["fps_max"]                   = -1
	# pose;
	params['number_people_max']         = 1
	params["net_resolution"]            = "336x336"
	# hand;
	params["hand"]                      = True
	params["hand_net_resolution"]       = "328x328"
	# required for gesture recognition
	params['frame_flip']                = False
	params['keypoint_scale']            = 3
	params["frame_first"]               = 0
	params["frame_last"]                = 74 
	
	return params

def openpose_driver(signvideodirectory, path_X, path_Y):
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
		
		parser = argparse.ArgumentParser()
		args = parser.parse_known_args()
	
		# iterate through all the raw videos;
		temp = os.path.join(signvideodirectory, '*.mp4')
		print("temp: ", temp)
		#for src_path in sorted(glob.glob(temp)):
		print(os.listdir(signvideodirectory))
		for src_path in glob.glob(temp):
			print("src_path: ", src_path)
	
			# set up the parameters for openpose except for json and video path;
			params = PARAMS()
			# current video has not been processed; 
			if not (ftools.checksubstring(src_path, "checked")):
				# Input video path 
				print('Processing:', src_path + '\n')
				params["video"] = src_path
				print("creating a temp directory to store the .*json files\n")
				with tempfile.TemporaryDirectory() as json_path:
					
					# Check that path creation is correct
					print('Writing JSON to a temporary directory:  ', json_path + '\n')
		
					# Set them to save json + open pose output video to the following paths
					params["write_json"]  = json_path
					# Run Open Pose
					try:
						opWrapper = op.WrapperPython(3)
						opWrapper.configure(params)
						opWrapper.execute()
					except Exception as e:
						print(e)
						sys.exit(-1)

					# done processing; we have our video json files now;
					# convert them to one single txt format;
					# any extra processing on the extracted keypoints before saving to txt?
					func_list = [SYNTH.pass_keypoints, SYNTH.perturb_keypoints]
					for j in range(len(func_list)):
						# use another temporary storage;
						print("creating a temp directory to store txt\n")
						with tempfile.TemporaryDirectory() as txt_path:
							filename = (((src_path.split('\\')[-1])).split('.'))[0]
							dummy_path = os.path.join(txt_path, filename + ".txt")
							# create the file within the temp directory;
							ftools.check_newfile(dummy_path)
							
							# now, all the file handling has been setlled;
							# process it;
							jv2t.json_video2txt(json_path, dummy_path, func_list[j])

							# json has been converted to txt;
							# to append the result to the parent txt files;

							# safeguard; ensure the files exist;
							print("Checking if the X.txt and Y.txt exist ...\n")
							if not os.path.exists(path_X):
								try:
									open(path_X, 'a').close()
								except Exception as e:
									print("An error occured", e)
									sys.exit(-1)
							if not os.path.exists(path_Y):
								try:
									open(path_Y, 'a').close()
								except Exception as e:
									print("An error occured", e)
									sys.exit(-1)
							# now save it to the label file, X.txt; Y.txt
							genxy.generate_XY(dummy_path, path_X,  path_Y)
							print("individual {X,Y} has been appended\n")

				# updating the number of processed video of this class; 
				print('now, updating the number of processed video of this class;\n')
				saved_counter = ftools.track_count(src_path, saved_path = "", save_name = 'saved_counter_train.p')
				print(saved_counter)
				print('\n')
				# sign off;
				# so that the processed video will not be processed again;
				print('the current video has been processed\n')
			
			# have already been processed;
			else:       
				print("the current video has already been processed, skip\n")
		# done;
		# outside of the outermost loop;		
		print('all videos in the folder have been processed at\n', signvideodirectory)
		# success?
		return True

	except Exception as e:
		print(e)
		sys.exit(-1)

# test driver;
if __name__ == '__main__':
	signvideodirectory = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\DUMMY\\HOSPITAL"
	path_X = os.path.join(signvideodirectory, "X_dummy.txt")
	path_Y = os.path.join(signvideodirectory, "Y_dummy.txt")
	openpose_driver(signvideodirectory, path_X, path_Y)