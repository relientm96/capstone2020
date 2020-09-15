#/usr/bin/env/python
# tools to (down) resample the frame in the form of txt for training;

# idea inspired by matthew;
# created by nebulaM78 team; capstone 2020;

import numpy as np
import random
import LSTM_tools as lstm
TEST_PATH = "C:\\Users\\yongw4\\Desktop\\X_train.txt"
# global constants;
FRAMES = 75
CUT_FRAMES = 70
FEATURE = 98
HALF = CUT_FRAMES/2

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
def split_half(arr):
	'''
	arg:
		np array of dimension (70, 98)
	return:
		a tuple of two sub-sampled arrays;
	function:
		sub-sample the array at odd and even;
	'''
	return (arr[0::2], arr[1::2]) 

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
	diff = meet_size - nrows
	
	# no point proceed;
	assert diff>0, "the number of rows is larger than the specified size"
	
	init = np.zeros((diff, ncol), dtype=np.float32)
	# append at the end;
	output = np.vstack([arr, init])
	assert output.shape[0] == meet_size, "makeup: np dimension should be different"
	return(output) 

def random_sample(arr, size = 35):
	'''
	arg:
		np array;
		meet_size; the required size;
	return:
		a randomly sampled array  ;
	'''

	# get the number of rows;
	nrows = arr.shape[0]
	print(nrows)
	assert nrows > 0, "make sure the array is non-empty!!"
	# generate a list of random numbers within a range;
	# this function makes sure there's no replacement, which is important;
	# (IMPORTANT!!) sort it 
	numbers = sorted(random.sample(range(0, nrows), size))
	
	idx = np.array(numbers)
	# now, randomly sample the frames;
	output = arr[idx,:]
	assert output[0] == meet_size, "random sampler: np dimension should be different"
	return output

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
def down_sampling(arr):
	if (arr.shape[0] == 75):
		arr = strip_five(arr)

	arr = remove_zero_rows(arr)
	shape = arr.shape[1]
	if(shape == 70):
		 first_half, second_half = split_half(arr)
		 assert (first_half[0], second_half[0]) == (35, 35), "np dimension should be (35, 35)"
		 return(first_half, second_half)
	else:
		if(int(shape/2) >= 30):
			first_half, second_half = split_half(arr)
			first_half = make_up(first_half, meet_size = 35)
			second_half = make_up(second_half, meet_size = 35)
			return(first_half, second_half)
		else:
			first_half = random_sample(arr, size = 35)
			second_half = random_sample(arr, size = 35)
			return(first_half, second_half)

		



	

# test driver;
if __name__ == '__main__':

	# txt file with 69 rows;
	TEST_PATH_03 = "C:\\Users\\yongw4\\Desktop\\X_train_03.txt"
	arrx = load_test(TEST_PATH_03, 69)
	test = arrx[0]
	print(test.shape)

	first_half, second_half = down_sampling(test)
	print(first_half.shape)
	print(second_half.shape)



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
	# txt file with 69 rows;
	TEST_PATH_03 = "C:\\Users\\yongw4\\Desktop\\X_train_03.txt"
	arrx = load_test(TEST_PATH_03, 69)
	test = arrx[0]
	print(test.shape)
	
	
	#test = make_up(test, meet_size = 75)
	test = random_sample(test)
	print(test.shape)
	'''

