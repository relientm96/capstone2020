# python built-in modules;
import cv2
import time
import sys
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
import multiprocessing as mp
import tempfile
import shutil
import errno
from numpy import random

# self-written modules;
import video_tools as VID
import file_tools as ftools
import openpose_setup as OP_SET

# global var;
DEGREE = [0, 3, 5, 7, 9, -3, -5, -7, -9]

	
def assign_video_locations(input_list, prefix, classname, identity, henshin_type):
	'''
	function:
		- auxiliary function;
		- to assign "IP input address" for the video transformation;
	args:
		- input_list; the list of parameters;
		- prefix;   the directory path;
		- classname; which class the current video being processed belongs to?
	return:
		- a list of the "IP addresses"
	'''
	length = len(input_list)
	# video format; hardcode this;
	vformat = ".mp4"
	temp = os.path.join(prefix, classname)
	locations = []
	for i in range(length):
		filename = temp + "_" + identity + "_" + henshin_type + str(i) + vformat
		ftools.check_newfile(filename)
		locations.append(filename)
	return locations

def which_block(func, input, storage_path):
	print("\n function: ", func)
	func(input, storage_path)

def rotate_block(input, storage_path):
	print("\n video input: ", input)
	# extract the info from the input pathname;
	tmp = input.split('.')
	prefix = tmp[0]
	token = (prefix.split('\\'))[-1]
	suffix = token.split('_')
	classname = suffix[0]
	speedname = suffix[1]

	# rotation degree;
	#DEGREE = [0, 3, 5, 7, 9, -3, -5, -7, -9]
	
	# assigning addresses for each rotation for convenienece;
	addresses = assign_video_locations(DEGREE, storage_path, classname, speedname, "r")
	#print('\n assigned address: ', addresses)
	num_workers = mp.cpu_count()  
	#print("number of cpu cores: ", num_workers)
	pool = mp.Pool(num_workers)
	for index in range(len(DEGREE)):
		print(index)
		pool.apply_async(VID.video_rotate, args=(input, addresses[index], DEGREE[index]))
	print('pooled')
	pool.close()
	pool.join()
	# end of function;

def flip_and_rotate_block(input, storage_path):
	# extract the info from the input pathname;
	tmp = input.split('.')
	prefix = tmp[0]
	token = (prefix.split('\\'))[-1]
	suffix = token.split('_')
	classname = suffix[0]
	speedname = suffix[1]

	# flip first;
	# create a temporary directory;
	with tempfile.TemporaryDirectory() as prefix:
		output_flip_path = os.path.join(prefix , "flipped_tmp.mp4")
		# create the file within the temp directory;
		ftools.check_newfile(output_flip_path)

		# the dummy file has been created;
		# safe to process;
		VID.video_flip(input, output_flip_path)
	
		# rotation degree;
		#DEGREE = [0, 3, 5, 7, 9, -3, -5, -7, -9]

		# assigning addresses for each rotation for convenienece;
		addresses = assign_video_locations(DEGREE, storage_path, classname, speedname, "fr")
		num_workers = mp.cpu_count()  
		pool = mp.Pool(num_workers)
		for index in range(len(DEGREE)):
			pool.apply_async(VID.video_rotate, args=(output_flip_path, addresses[index], DEGREE[index]))
		pool.close()
		pool.join()
		# end of function;

# exception handler  
# needed for synthesize_bloack() function below;
def handler(func, path, exc_info):  
	print("Inside handler")  
	print(exc_info)

def synthesize_block(input, speed_seed, main_X, main_Y):
	try:
		# create temporary locations
		tmp_dir = tempfile.mkdtemp()
		x1_path = os.path.join(tmp_dir, "x1.txt")
		y1_path = os.path.join(tmp_dir, "y1.txt")
		# create the files within the temp directory;
		ftools.check_newfile(x1_path)
		ftools.check_newfile(y1_path)

		tmp_dir2 = tempfile.mkdtemp()
		x2_path = os.path.join(tmp_dir, "x2.txt")
		y2_path = os.path.join(tmp_dir, "y2.txt")
		# create the files within the temp directory;
		ftools.check_newfile(x2_path)
		ftools.check_newfile(y2_path)

		storage_paths = [tmp_dir, tmp_dir2]
		paths_X = [x1_path, x2_path]
		paths_Y = [y1_path, y2_path]
		
		# get the classname of the video;
		[classname, _] = ftools.checkoff_file(input, "")
		# create a temporary directory;
		with tempfile.TemporaryDirectory() as prefix:
			identity = "s" + str(int(speed_seed*10))
			output_speed_path = prefix + "\\" + classname + "_" + identity + ".mp4"
			# make sure the file in the temp exists;
			ftools.check_newfile(output_speed_path)
		
			VID.video_speed(input, output_speed_path, speed_seed)
		
			# now, pass through:
			# transformation 1: only rotate;
			# transformation 2: flip then rotate;
			rotate_block(output_speed_path, storage_paths[0])
			flip_and_rotate_block(output_speed_path, storage_paths[1])
		
		# maximize the cpu cores;
		num_workers = mp.cpu_count()  
		pool = mp.Pool(num_workers)
		print("entering the pool")
		for i in range(len(storage_paths)):
			pool.apply_async(OP_SET.openpose_driver, args=(storage_paths[i], paths_X[i], paths_Y[i]))
		print("pooled")
		pool.close()
		pool.join()

		# append the results to the main path_X.txt and path_Y.txt
		for i in range(len(paths_X)):
			ftools.append_file(paths_X[i], main_X)
			ftools.append_file(paths_Y[i], main_Y)

	# the created temporary paths are no longer needed;
	# clean up;
	finally:
		try:
			# delete directory
			shutil.rmtree(tmp_dir, onerror = handler)  
			shutil.rmtree(tmp_dir2, onerror = handler)
		except OSError as exc:
			 # ENOENT - no such file or directory
			if exc.errno != errno.ENOENT: 
				# re-raise exception
				raise  

#----------------------------------------------------------------------
# here, we synthesize by augmenting the keypoints
# instead on the raw video itself;
#----------------------------------------------------------------------
# a neutral function
# the output is equal to the input;
def pass_keypoints(input_list):
	return input_list

# perturb the keypoints;
def perturb_keypoints(input_list):
	# loc = mean;
	# scale = standard deviation
	noise = random.normal(loc = 0, scale = 0.003)
	noise2 = random.normal(loc = 0, scale = 0.003)
	ls = input_list
	length = len(input_list)
	fixed = 2
	for index, elem in enumerate(input_list):
		# the index for the confidence;
		if ((abs(index - fixed)%3) == 0):
			xval = input_list[index -1]
			yval = input_list[index -2]
			# since we have normalized the keypoints;
			# we dont want the added noise to "violate" the range;
			if not (abs(xval) == 1):
				ls[index-1] = input_list[index -1] + noise
			if not (abs(yval) == 1):
				ls[index-2] = input_list[index -2] + noise2
	return ls

	
if __name__ == '__main__':

	path_X = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\X_dummy.txt"
	path_Y = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\Y_dummy.txt"

	# safeguard;
	# create them if the files do not exist;
	print("Checking if the X.txt and Y.txt exist ...\n")
	if not (os.path.exists(path_X) or os.path.exists(path_Y)):
		try:
			open(path_X, 'w').close()
			open(path_Y, 'w').close()
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)
	input = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_14.mp4"
	speed_seed = 2
	synthesize_block(input, speed_seed, path_X, path_Y)
	#encapsulate_block(input, speed_seed, path_X, path_Y)













