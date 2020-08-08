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

def synthesize_block(input, speed_seed, path_X, path_Y):
	
	# create a temp direc;
	with tempfile.TemporaryDirectory() as storage_path:
		# create two sub directories wrt storage_path;
		# to store two different sets;
		# so that we could run openpose parallely on the two sets;
		storage_path_01 = os.path.join(storage_path, "temp_01")
		storage_path_02 = os.path.join(storage_path, "temp_02")

		storage_path_01 = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\temp_01"
		storage_path_02 = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\temp_02"
		# get the classname of the video;
		[classname, _] = ftools.checkoff_file(input, "")
		# create a temporary directory;
		with tempfile.TemporaryDirectory() as prefix:
			identity = "s" + str(int(speed_seed*10))
			output_speed_path = prefix + "\\" + classname + "_" + identity + ".mp4"
			VID.video_speed(input, output_speed_path, speed_seed)
		
			# now, pass through:
			# transformation 1: only rotate;
			# transformation 2: flip then rotate;
			rotate_block(output_speed_path, storage_path_01)
			flip_and_rotate_block(output_speed_path, storage_path_02)
		
			direc_list = [storage_path_01, storage_path_02]
			print("list: ", direc_list)

			OP_SET.openpose_driver(storage_path_01, path_X, path_Y)
			'''
			# maximize the cpu cores;
			num_workers = mp.cpu_count()  
			pool = mp.Pool(num_workers)
			print("entering the pool")
			for direc in direc_list[0]:
				pool.apply_async(OP_SET.openpose_driver, args=(direc, path_X, path_Y))
			print("pooled")
			pool.close()
			pool.join()
			'''
if __name__ == '__main__':

	path_X = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\X_dummy.txt"
	path_Y = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\Y_dummy.txt"
	input = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_14.mp4"
	speed_seed = 2
	synthesize_block(input, speed_seed, path_X, path_Y)














