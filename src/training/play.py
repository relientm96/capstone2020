#from skimage import io
#from skimage import transform as tf
import cv2
import numpy as np

#import tensorflow as tf

import os
from matplotlib import image
import matplotlib.pyplot as plt

path = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-images\\test1.jpg"

path = "C:\\Users\\yongw4\\Desktop\\test"
# to count the number of subdirectories at the top layer;
def subdir_count(path):
	count = 0
	for root, dirs, files in os.walk(path):
		for i in range(len(dirs)):
			classvideo = os.path.join(root, dirs[i])
			print(classvideo)
		break
	
# test driver;
if __name__ == '__main__':
	subdir_count(path)


    
def transform_block(func, input, storage_path, DEGREE):
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
		pool.apply_async(func, args=(input, addresses[index], DEGREE[index]))
	print('pooled')
	pool.close()
	pool.join()
	# end of function;

def flip_and_transform_block(func, input, storage_path, DEGREE):
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
			pool.apply_async(func, args=(output_flip_path, addresses[index], DEGREE[index]))
		pool.close()
		pool.join()
		# end of function;