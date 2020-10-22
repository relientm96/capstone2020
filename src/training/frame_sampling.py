#/usr/bin/env/python
# tools to (down) resample the frame in the form of txt for training;

# idea inspired by matthew;
# created by nebulaM78 team; capstone 2020;

import numpy as np
import random
from glob import glob
import sys, os
import multiprocessing as mp
import LSTM_tools as lstm
import file_tools as ft

# global constants;
FRAMES = 75
CUT_FRAMES = 70
FEATURE = 98
HALF = int(CUT_FRAMES/2)

# write a np array into txt;
def save_arr(array, filepath):
	with open(filepath, 'ab') as fp:
		np.save()

# for unit testing
def load_test(X_path, n_steps):
	
	file = open(X_path, 'r')
	X_ = np.array(
		[elem for elem in [
			row.split(',') for row in file
		]], 
		dtype=np.float32
	)
	file.close()
	blocks = int(len(X_) / n_steps)
	X_ = np.array(np.split(X_,blocks))

	return X_ 

def strip_five(arr):
	'''
	arg:
		np array of dimension (75, 98)
	return:
		sliced np array;
	function:
		remove the first 2 rows, and the last 3 rows;
	'''
	# hardcode the constant dimensions;
	assert arr.shape == (FRAMES, FEATURE), "np dimension should be (75, 98)"
	return(arr[2:72])

# remove all the zero frames;
def remove_zero_rows(arr):
	'''
	arg:
		np array of dimension (75, 98)
	return:
		np array with all the zero rows removed;
	'''
	return(arr[~np.all(arr == 0, axis=1)])

# this assumes the dimension = (70, 95);
def split_half(arr, meet_size = HALF):
	'''
	arg:
		np array of dimension (70, 98)
	return:
		a tuple of two sub-sampled arrays;
	function:
		sub-sample the array at odd and even;
	'''

	# split themm
	out1 = arr[0::2]
	out2 = arr[1::2]
	return(out1, out2)

def make_up(arr, meet_size = 35):
	'''
	arg:
		np array;
		meet_size; the required size;
	return:
		a filled-up np array;    
	function:
		to fill up the array to meet a size requirement;
	'''

	# get the number of rows and columns;
	nrows = arr.shape[0]
	ncol = arr.shape[1]
	#print('nrows: ', nrows)
	diff = meet_size - nrows
	
	# no point proceed;
	assert diff>=0, "the number of rows is larger than the specified size"
	
	init = np.zeros((diff, ncol), dtype=np.float32)
	# append at the end;
	output = np.vstack([arr, init])
	assert output.shape[0] == meet_size, "makeup: np dimension should be different"

	return(output)

def random_sample(arr, meet_size = HALF):
	'''
	arg:
		np array;
		meet_size; the required size;
	return:
		a randomly sampled array  ;
	'''

	# get the number of rows;
	nrows = arr.shape[0]
	#print(nrows)
	assert nrows > 0, "make sure the array is non-empty!!"
	# generate a list of random numbers within a range;
	# this function makes sure there's no replacement, which is important;
	# (IMPORTANT!!) sort it 
	numbers = sorted(random.sample(range(0, nrows), meet_size))
	
	idx = np.array(numbers)
	# now, randomly sample the frames;
	output = arr[idx,:]
	assert output.shape[0] == meet_size, "random sampler: np dimension should be different"

	return(output)

def reshape(arr, extra = [], meeting_size = HALF):
	'''
		function:
				return as np of dimension: (1,35,98) if there's only one arr passed;
				return as np of dimension: (2,35,98) if there are two arrs passed;
		
		args: arr; np array;
		extra: a list of one np array;
		meeting_size; dimension = 35;
	'''

	tmp = np.empty((0, meeting_size, FEATURE), dtype = np.float32)
	tmp = np.insert(tmp, 0 , arr, axis=0)
	
	# extra argument invoked;
	length = len(extra)
	#print(length)
	if(length > 0):
		assert length == 1, 'there should be only one np array as element'
		tmp = np.insert(tmp, 1 , extra[0], axis=0)
	# here, we have either (1,35,98) or (2,35,98) np array;
	return(tmp)
	

# execution order;
# 1. cut down the 75-frame txt to 70-frame;
# 2. remove all the zero-rows;
# 3. if the np.shape == 70, then split to 35-35;
# 4. else if np.shape < 70:
#       if np.shape/2 >= 30:
#           fill it up to 35 to have 35-35;
#       else:
#           too small to split it into 35-35;
#           randomly sample 35 from it twice to have two 35's;
def down_sampling(arr, meet_size = HALF):
	# step (1)
	if (arr.shape[0] == 75):
		arr = strip_five(arr)
	# step (2)
	arr = remove_zero_rows(arr)

	# get the number of rows;
	shape = arr.shape[0]
	#print(shape)
	
	# step (3)
	if(shape == 70):
		 output = split_half(arr)
		 return(reshape(output))
	# step (4)
	else:
		#print("shape: ", shape)
		if((int(shape/2) < 35) and (int(shape/2) >= 30)):
			first_half, second_half = split_half(arr)
			first_half = make_up(first_half, meet_size = 35)
			second_half = make_up(second_half, meet_size = 35)
			return(reshape(first_half, [second_half]))
		elif((int(shape/2) < 30) and (shape >=35)):
			first_half = random_sample(arr, meet_size = 35)
			second_half = random_sample(arr, meet_size = 35)
			return(reshape(first_half, [second_half]))
		else:
			# here, it must have an original size less than 30;
			# but concatenate it twice so that we have consistency throughout
			tmp = make_up(arr)
			return(reshape(tmp, [tmp]))

# process one txt file;
def process_one(filepath, format):
	print("entering process_one function")

	# initialize an empty np;
	output = np.empty((0, HALF, FEATURE), dtype = np.float32)

	# txt or numpy?
	if(format == "txt"):
		# load the training X files;
		X_load = lstm.load_X(filepath)
	elif(format == "npy"):
		X_load = np.load(filepath)

	nsample = X_load.shape[0]
	print('number of samples: ', nsample)
		
	# run through all the samples;
	i = 0
	print("to do down-sampling")
	while(i < nsample):
		# down sample the current 75-chunk;
		processed = down_sampling(X_load[i])
		#print("sanity check\n the processed chunk has a dimension of: ", processed.shape)
		# insert it into output;
		output = np.insert(output, 0 , processed, axis=0)
		i = i+1
	# end?
	#print(output.shape)
	print("done processing")
	return output

def gen_XY(rootpath):
	print("entering gen_xy")
	loc = os.path.join(rootpath, "*.txt")
	for txt in sorted(glob(loc)):
		print(txt)
		# just in case evn though it has been sorted ...
		fname = txt.split("\\")[-1]
		tmpname = fname.split(".")[0]
		#print('tmpname: ', tmpname)
		fname = fname.split("_")[0]
		#print("fname: ", fname)
		low = fname.lower()
		# training X file;
		if (low == "x"):
			outputX = process_one(txt, "txt")
			nsample = outputX.shape[0]
			savename = os.path.join(rootpath, tmpname+"_down.npy")
			np.save(savename, outputX)
			print("outputX size: ", outputX.shape)
		# the label file;
		elif (low == "y"):
			gety = lstm.load_Y(txt)
			classname = gety[0][0]
			#print("classname: ", classname)
			savename = os.path.join(rootpath, tmpname + "_down.npy")
			# initialize new array for the Y;
			outputY = np.empty((nsample, 1), dtype = np.int8)
			outputY.fill(int(classname))
			np.save(savename, outputY)
			print("outputY size: ", outputY.shape)
		else:
			return None

def process_block(directory_path):
	ls = []
	for root, dirs, files in os.walk(directory_path, topdown=False):
		print('entering')
		print("root: ", root)
		#loc = os.path.join(root, "*.txt")
		ls.append(root)
		print(ls)
	print("ls: ", ls)

	# use multiprocessing to process all the txt files;
	num_workers = mp.cpu_count()
	print("cpu workers: ", num_workers)
	pool = mp.Pool(num_workers)
	for index in range(0, len(ls)):
		print("entering the pool?")
		print(ls[index])
		pool.apply_async(gen_XY, args=(ls[index],))
	pool.close()
	pool.join()

# test driver;
if __name__ == '__main__':
	prefix = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train-21-10-2020\\text-dataset\\shear"
	process_block(prefix)
	'''
	# get the 75-frame training data;
	X_75 = np.load(prefix+"\\X_combine.npy")
	Y_75 = np.load(prefix+"\\Y_combine.npy")

	# reduce the samples;
	X_35 = process_one(prefix+"\\X_combine.npy", "npy")
	Y_35 = process_one(prefix+"\\Y_combine.npy", "npy")
	
	# sanity check;
	print("x_75 shape: ", X_75)
	print("y_75 shape: ", Y_75)

	print("x_35 shape: ", X_35)
	print("y_35 shape: ", Y_35)
	'''
	# save them;
	#prefix = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train-21-10-2020\\train-npy"
	#np.save(prefix+"\\X_train_35.npy", X_35)
	#np.save(prefix+"\\Y_train_35.npy", Y_35)
	
	#directory_path = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train"
	#TEST_PATH = "C:\\Users\\yongw4\\Desktop\\down-sampling\\X_train.txt"
	#process_block(directory_path)
	#filepath = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train\\speed_08\\AMBULANCE\\Y_AMBULANCE_train_down.npy"
	#y = np.load(filepath)
	#print(y.shape)
	
	#TEST_PATH = "C:\\Users\\yongw4\\Desktop\\down-sampling\\X_train.txt"
	#process_one(TEST_PATH)
	'''
	# txt file with 69 rows;
	TEST_PATH_03 = "C:\\Users\\yongw4\\Desktop\\down-sampling\\X_train_04.txt"
	arrx = load_test(TEST_PATH_03, 58)
	test = arrx[0]
	print(test.shape)

	dum = down_sampling(test)
	print(dum.shape)
	'''
	'''
	print(first_half.shape)
	print(second_half.shape)
	dum = reshape(first_half, [second_half])
	print(dum.shape)
	
	#prefix = "C:\\Users\\yongw4\\Desktop\\down-sampling"
	#writep = prefix+"\\test123.txt"
	#write2text(first_half, writep)
	'''
	'''
	ft.npy2text(first_half, writep)
	ft.npy2text(second_half, writep)

	prefix = "C:\\Users\\yongw4\\Desktop\\down-sampling"
	writep = prefix+"\\first.txt"
	ft.npy2text(first_half, writep)
	
	prefix = "C:\\Users\\yongw4\\Desktop\\down-sampling"
	writep = prefix+"\\second.txt"
	ft.npy2text(second_half, writep)

	'''

	'''
	# txt file with alot of zero rows;
	TEST_PATH_02 = "C:\\Users\\yongw4\\Desktop\\X_train_02.txt"
	arrx = lstm.load_X(TEST_PATH_02)
	print(arrx.shape)
	test = arrx[0]
	print(test.shape)
	test = remove_zero_rows(strip_five(test))
	print(test.shape)
	'''
	
	'''
	naked = strip_five(test)
	print(naked.shape)

	(sub01, sub02) = sub_sample(naked)
	print(sub01.shape)
	print(sub02.shape)
	'''
	'''
	TEST_PATH_03 = "C:\\Users\\yongw4\\Desktop\\down-sampling\\X_train_03.txt"
	arrx = load_test(TEST_PATH_03, 69)
	test = arrx[0]
	print(test.shape)
	
	#test = make_up(test, meet_size = 80)
	#test = strip_five(test)
	#test = random_sample(test)
	
	#test = split_half(test)
	print(test.shape)
	'''
	
