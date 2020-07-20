#/usr/bin/env/python
# created by openpose team;
# to generate X, Y.txt files from a database for training;
# modified by matthew; nebulaM78 team; capstone 2020;

''' pipeline: preparing video dataset for training; 
1. one raw video -> tmp*.json --> tmp.txt --> {X.txt, Y.txt}
note: the filename of the processed raw video will be changed to indicate that
	it has been processed so that it will not be run again;
'''

'''
Module to run openpose with set configurations over all input video files in a directory
--> generate its corresponding X, Y training and labels, respectively into files at: "path_X", "path_Y"
'''

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

# from nebulaM78;
import file_tools as ftools
import generate_XY as genxy
import json_video2txt as jv2t

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

	#------------------------------------------------------------
	# PARAMETERS;
	# Static open pose configurations
	## Configurations for openpose (subject to change) ##
	# More info for these parameters here:
	# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/b5bffe18a8021f5f3ed98f19441b658647d9a8c3/include/openpose/flags.hpp
	## note: you can also look into rotating frames ##
	#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/b5bffe18a8021f5f3ed98f19441b658647d9a8c3/include/openpose/flags.hpp#L51

	#------------------------------------------------------------
	params = dict()

	# Turn off display for faster processing
	params["display"]             = 0
	# Model folder path
	params["model_folder"]        = "../openpose-python/models/"
	# Turn on hand recognition
	params["hand"]                = True
	# Pose net resolution
	params["net_resolution"]      = "336x336"
	# Hand Net Resolution
	params["hand_net_resolution"] = "328x328"
	# Type of pose model
	params["model_pose"]          = "BODY_25"
	# normalize the keypoints to [0,1]
	params['keypoint_scale']      = 3

	# Define which is the starting frame and the last frame; for open pose to process
	params["frame_first"]         = 0
	params["frame_last"]          = 74  # due to hardware limitation;
	
	# Limit to how many FPS to run Openpose on
	params["fps_max"]             = 8    

	# we just need json as long as the video is within the "limitation" of openpose;
	# turn off the unnecessary parameters;
	params['render_pose']             = 0
	params['display']                 = 0

	# by definition, openpose will keep the one with highest score (probabiltity)
	# this will remove the false positives;
	params['number_people_max']     = 1

	#-----------------------------------------------------
	# CHANGE HERE!!!
	# training GLOBAL constants;
	#-----------------------------------------------------
	# our training set and its label;
	path_X = "C:\\Users\\yongw4\\Desktop\\auslan-test-videos\\X.txt"
	path_Y = "C:\\Users\\yongw4\\Desktop\\auslan-test-videos\\Y.txt"
	# raw video source;
	signvideodirectory = "C:\\Users\\yongw4\\Videos\\basic\\dummy"

	# iterate through all the raw videos;
	for input_vid_path in sorted(glob.glob(os.path.join(signvideodirectory + '\\*.mp4'))):
		#------------------------------------------------------
		# checking if the filename is named correctly;
		# if not, skip it;
		# take a note and save it to a txt;
		# case sensitive; so we have to follow certain form;
		#------------------------------------------------------
		# to update this list eventually;
		CLASSES = ['pain', 'ambulance']
		print(input_vid_path)
		tmp = (input_vid_path.split('\\'))[-1]
		tmp = (tmp.split('.'))[0]
		checkname = tmp.split('_')[0]
		#print(checkname)
	
		# correctly name; process it;
		if (checkname in CLASSES):
			# current video has not been processed; 
			if not (ftools.checksubstring(input_vid_path, "checked")):
				# Input video path 
				print('Processing:', input_vid_path)
				# Add to params for open pose to use
				params["video"] = input_vid_path
		
				print("creating a temp directory to store the .*json files\n")
				with tempfile.TemporaryDirectory() as json_path:
					# Check that path creation is correct
					print('Writing JSON to a temporary directory:  ', json_path)
		
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
					# by using another temporary storage;
					print("creating a temp directory to store txt\n")
					with tempfile.TemporaryDirectory() as txt_path:
						filename = (((input_vid_path.split('\\')[-1])).split('.'))[0]
						dummy_path = txt_path + "\\" + filename + ".txt"
						jv2t.json_video2txt(json_path, dummy_path)
				
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
				ftools.track_count(src_path, saved_path = "", save_name = 'saved_counter.p')
			
				# sign off;
				# so that the processed video will not be processed again;
				print('the current video has been processed\n')
				[classname, dst_checked] = ftools.checkoff_file(input_vid_path, "checked")
				os.rename(input_vid_path, dst_checked)
			
			# have processed;
			else:
				print("the current video has already been processed, skip\n")
		
		# the filename is not proper;
		# to-do; shall automatically spellcheck and correct it;
		else:
			# save the note at the current directory
			path_M = "TAKE_NOTE.txt"
			with open(path_M, 'a') as file:
				file.write(input_vid_path)
				print("the video filename is improper: \n", input_vid_path)
				print("check the note at: ", os.getcwd()+ "\\" + path_M)
		break

except Exception as e:
	print(e)
	sys.exit(-1)