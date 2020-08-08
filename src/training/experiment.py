import cv2
import time
import sys
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
import file_tools as ftools
import multiprocessing as mp
import video_tools as VID
import multi_check as MC
import tempfile
import shutil
import errno

# src - https://stackoverflow.com/questions/48191238/can-multiple-processes-write-to-the-same-folder

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
	print('\n assigned address: ', addresses)
	num_workers = mp.cpu_count()  
	print("number of cpu cores: ", num_workers)
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
		output_flip_path = os.path.join(prefix , "tmp.mp4")
		VID.video_speed(input, output_flip_path)
	
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

def synthesize_block(input, speed_seed, storage_path):
	# get the classname of the video;
	[classname, _] = ftools.checkoff_file(input, "")
	print("classname: ", classname)
	# create a temporary directory;
	with tempfile.TemporaryDirectory() as prefix:
		identity = "s" + str(int(speed_seed*10))
		output_speed_path = prefix + "\\" + classname + "_" + identity + ".mp4"
		VID.video_speed(input, output_speed_path, speed_seed)

		# now, pass through:
		# transformation 1: only rotate;
		# transformation 2: flip then rotate;
		transform = [rotate_block, flip_and_rotate_block]
		#rotate_block(output_speed_path, storage_path)
		num_workers = mp.cpu_count()  
		pool = mp.Pool(2)
		print("entering the pool")
		for func in transform:
			pool.apply_async(which_block, args=(func, output_speed_path, storage_path))
		print("pooled")
		pool.close()
		pool.join()
		

if __name__ == '__main__':

	storage_path = "C:\\Users\\yongw4\\Desktop\\test_synthesis"
	input = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\help_95.mp4"
	speed_seed = 0.7
	synthesize_block(input, speed_seed, storage_path)














