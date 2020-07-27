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
import ffmpeg

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
	path_X = "C:\\Users\\yongw4\\Desktop\\TREASURE\\X_test.txt"
	path_Y = "C:\\Users\\yongw4\\Desktop\\TREASURE\\Y_test.txt"
	
	# raw video source;
	signvideodirectory = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE\\"

	# iterate through all the raw videos;
	#for src_path in sorted(glob.glob(os.path.join(signvideodirectory + '\\*.mp4'))):
	for root, dirs, files in os.walk(signvideodirectory, topdown=False):
		for name in files:
			# make sure the current file is MP4;
			ext = name.split('.')[-1]
			if (ext != "mp4"):
				continue
			# OK, it's mp4; process it;
			src_path = os.path.join(root, name)
			
			# current video has not been processed; 
			if not (ftools.checksubstring(src_path, "checked")):
				# get the classname;
				[classname, _] = ftools.checkoff_file(src_path, "")
				with tempfile.TemporaryDirectory() as flipped_dummy_path:
					flipped_path = flipped_dummy_path + '\\' + classname + '_flipped.mp4'
					(ffmpeg
						.input(src_path)
						.hflip()
						.output(flipped_path)
						.global_args('-loglevel', 'error')
						.global_args('-y')
						.run()
					)
					print("the current video has been flipped and save at the temp directory: ", flipped_path + '\n')
					PATHS = [src_path, flipped_path]
					input_vid_path = PATHS[0]
					for input_vid_path in PATHS:
						# Input video path 
						print('Processing:', input_vid_path + '\n')
						params["video"] = input_vid_path
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
							# by using another temporary storage;
							print("creating a temp directory to store txt\n")
							with tempfile.TemporaryDirectory() as txt_path:
								#print("DEBUGGING\n")
								filename = (((input_vid_path.split('\\')[-1])).split('.'))[0]
								dummy_path = txt_path + "\\" + filename + ".txt"
				
								#print("DEBUGGING-01\n")
				
								jv2t.json_video2txt(json_path, dummy_path)
								#print("DEBUGGING-02\n")
				
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
				print('here, the current video and its corresponding flipped video have been processed\n')
				print('now, updating the number of processed video of this class;\n')
				saved_counter = ftools.track_count(src_path, saved_path = "", save_name = 'saved_counter_train.p')
				print(saved_counter)
				print('\n')
				# sign off;
				# so that the processed video will not be processed again;
				print('the current video has been processed\n')
				[classname, dst_checked] = ftools.checkoff_file(src_path, "checked")
				os.rename(src_path, dst_checked)
			
			# have processed;
			else:       
				print("the current video has already been processed, skip\n")
		
	# outside of the outermost loop;		
	print('all videos in the folder have been processed at\n', signvideodirectory)
except Exception as e:
	print(e)
	sys.exit(-1)