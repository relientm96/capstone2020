#/usr/bin/env/python
# auxiliary tool set for file checking purposes;
# created by nebulaM78 team; capstone 2020;

import numpy as np
from pprint import pprint
import glob, os, sys
import random
import operator
# needed to save non-string python object; eg dictionary;
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle
import LSTM_tools as lstm

# functions:
# 1. update_dict()
# 2. check_new_key()
# 3. create_dict()


# to rename a file name and obtain the classname;
def checkoff_file(src, add_vocab):
	'''
	args - 
		1. src - (string) the original path;
		2. add_vocab - (string) the word to append
	return - a list of 
		1. the modified path name
		2. the classname
	example:
		src = C:\\capstone\\a_1.txt;
		vocab = checked;
		--> return( C:\\capstone\\a_1_checked.txt, a)
	'''
	print("checkoff_file; src:", src)
	tmp = src.split('.')
	prefix = tmp[0]
	token = (prefix.split('\\'))[-1]
	classname = token.split('_')[0]
	format = tmp[-1]
	dst_checked = prefix + "_" + add_vocab + "." + format
	#print("checked\n", dst_checked)
	#print("classname\n", classname)
	return [classname, dst_checked]

# check for substring
def checksubstring(src, substring):
	'''
	args - 
		1. src - (string) the original path;
		2. substring - (string) to be checked against
	return -
		True or False   
	'''
	tmp = src.split('.')
	prefix = tmp[0]
	token = (prefix.split('\\'))[-1]
	return (substring in token)

def track_count(src_path, saved_path = "", save_name = 'saved_counter.p'):
	'''
	function
		- create/update a counter to track the progress in processing the dataset;
	args 
		1. src_path; 
		2. saved_path; where to save for the dictionary;
		3. save_name; the name of the dictionary to save;
	return 
		- counter (dictionary)
	'''
	# check whether we have an existing dictionary;
	# src - https://stackoverflow.com/questions/28633555/how-to-handle-filenotfounderror-when-try-except-ioerror-does-not-catch-it
	#to load the saved dictonary;
	dict_path = saved_path + save_name

	# get the class name;
	tmpstring = (src_path.split('\\'))[-1]
	tmpstring = (tmpstring.split('.'))[0]
	classname = (tmpstring.split('_'))[0]
	 
	try:
	  with open(dict_path, 'rb') as fp:
		  dict = pickle.load(fp)
		  print('the existing counter has been loaded\n', dict)
		  # existing key? update the count;
		  if (classname in dict.keys()):
			  print("updating the count\n")
			  dict[classname] = dict[classname] + 1
		  #initialize it;
		  else:
			  print('new entry\n')
			  dict[classname] = 1
	except OSError as e:
		 print("e\n Need to create new counter\n")
		 dict = {}
		 dict[classname] = 1
	
	# done everything? save it then;
	with open(dict_path, 'wb') as fp:
		pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("the counter has been saved;\n")

	return dict

def append_file(src, dst):
	with open(dst, 'a') as fileX, open(src, 'r') as fileY:
		print('given the source\n', src)
		print("Appending ... \n")
		for line in fileY:
			# safe guard;
			assert(line != "")
			#print("txt line, \n", line)
			fileX.write(line)
		print("finished appending\n")

# create the file within the temp directory;
def check_newfile(file_path):
	if not (os.path.exists(file_path)):
		try:
			open(file_path, 'w').close()
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)

# to count the number of subdirectories at the top layer;
def subdir_count(path):
	count = 0
	for root, dirs, files in os.walk(path):
		print('dirs: ', dirs)
		count += len(dirs)
		break
	return count

def get_class_dict_info(filedirectory, search_term):
	'''
		args: 
			filedirectory; where are the text files?
			search_term; txt format or npy format;
		return:
			the class info with the lowest number of samples;
		function:
			to find out which class has the lowest number of samples;
	'''
	dict = {}
	# how many set it has?
	ncombine = subdir_count(filedirectory)
	print("ncombine: ", ncombine)
	sys.exit('DEBUG')
	if not(os.path.isdir(filedirectory)):
		sys.exit("the directory is not valid;")
	print('directory: ', filedirectory)
	for root, dirs, files in os.walk(filedirectory, topdown=False):
		# need to make sure the file is "upper-capitalized";
		loc = os.path.join(root, "Y*."+search_term)
		for search_file in sorted(glob.glob(loc)):
			print(search_file)
			# txt format;
			if(search_term == "txt"):
				dataset = lstm.load_Y(search_file)	
				classname = dataset[0][0]
				# get the number of rows;
				try:
					dict[classname] = dict[classname] +  np.size(dataset, 0)
				except KeyError as e:
					print('there is an error: %s, so ... do it differently', e)
					dict[classname] = np.size(dataset, 0)
			# in npy format; 
			else:
				nparray = np.load(search_file)
				classname = nparray[0][0]
				try:
					dict[classname] = dict[classname] + nparray.shape[0]
				except KeyError as e:
					print('there is an error: %s, so ... do it differently', e)
					dict[classname] = nparray.shape[0]
				
	print(dict)
	# get the min;
	minimum = min(dict.items(), key=operator.itemgetter(1))
	print('class with the lowest samples: ', minimum)
	#classmin = minimum[0]
	#min_num = minimum[1]
	#ls = [(key, value) for key,value in dict.items() if key != classmin]
	return (ncombine, minimum)

def balance_data_sample(dataset, sample_size):
	'''
		args:
			- dataset; the numpy array;
			- sample_size;
		return:
			- the "downsampled" array;
		function:
			- to downsample the class samples to sample_size;
	'''
	nrows = np.size(dataset, 0)
	# get a list of unique random int within the sample size;
	ls = random.sample(range(0, nrows), sample_size)
	down_arr = dataset[ls,:,:]
	return down_arr


# combine all the np arrays;
def patch_nparrays(txt_directory, search_term):
	'''
		task:
			load all the generated txt files as np arrays and combine into one;
		args:
			txt_directory; the directory where all the txt files are stored;
			search_term; txt format or npy format;
		returns:
			a tuple of (X, Y)
	'''
	# get all the classes distribution info;
	# and return the one with the lowest samples;
	(ncombine, minimum) = get_class_dict_info(txt_directory, search_term)
	sample_size = minimum[1]

	# now, patch all the samples across the classes;
	PATCH = [[],[]]
	for root, dirs, files in os.walk(txt_directory, topdown=False):
		loc = os.path.join(root, "*."+search_term)
		for search_file in sorted(glob.glob(loc)):
			# X or Y?
			tmp = search_file.split("\\")[-1]
			tmp = tmp.split("_")[0].lower()
			
			proportion = int(sample_size/ncombine)
			#proportion = int(sample_size/2)
			# txt or npy? process them differently;
			if(search_term == "txt"):
				# the training file, x
				if(tmp == "x"):
					# just an indicator;
					INDEX = 0
					# load it;
					dataset = lstm.load_X(search_file)
					print('prior size: ', dataset.shape)
					# handle imbalanced distribution, if any;
					dataset = balance_data_sample(dataset, proportion)
					print("after size: ", dataset.shape)
				# the label file, Y;
				else:
					INDEX = 1
					# handle imbalanced dist, if any
					dataset = lstm.load_Y(search_file)
					print("prior size: ", dataset.shape)
					dataset = dataset[0:proportion,:]
					print("after size: ", dataset.shape)
			# npy files;
			else:
				if(tmp == "x"):
					INDEX = 0
					dataset = np.load(search_file)
					print('prior size: ', dataset.shape)
					# handle imbalanced distribution, if any;
					dataset = balance_data_sample(dataset, proportion)
					print("after size: ", dataset.shape)
				else:
					INDEX = 1
					dataset = np.load(search_file)
					print('prior size: ', dataset.shape)
					# handle imbalanced distribution, if any;
					dataset = dataset[0:proportion,:]
					print("after size: ", dataset.shape)
			# done?
			PATCH[INDEX].append(dataset)
	# concatenate all the arrays into one;
	X_monstar = np.concatenate(tuple(PATCH[0]), axis = 0)
	Y_monstar = np.concatenate(tuple(PATCH[1]), axis = 0)
	return (X_monstar, Y_monstar)

# save and load the huge numpy array in npz format;
def npy_write(data, filename):
	np.save(filename, data)

# read np array saved in npz;
def npy_read(path):
	return np.load(path)

# load a saved np array in npz format and svae as text;
def npy2text(array, filepath):
	np2file = open(filepath, 'a+')
	for row in np.load(array):
		np.savetxt(np2file, row)
	np2file.close()

# write a np array into txt;
def write2text(array, filepath):
	np2file = open(filepath, 'a+')
	for row in array:
		print(row)
		sys.exit("debug")
		#np.savetxt(np2file, row)
	np2file.close()


# test driver;
if __name__ == '__main__':
	
	prefix = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\train"
	#sign_dir = prefix+"\\4-hospital-txt\\X_train.txt"
	#(x_mon, y_mon)= patch_nparrays(prefix, "txt")
	#print(x_mon.shape, y_mon.shape)

	(x_mon, y_mon)= patch_nparrays(prefix, "npy")
	print(x_mon.shape, y_mon.shape)

	filenameX = prefix+"\\X_main.npy"
	filenameY = prefix+"\\Y_main.npy"

	np.save(filenameX, x_mon)
	np.save(filenameY, y_mon)



	#get_class_dict_info(prefix, "npy")
	#np_X = lstm.load_X(sign_dir)
	#sample_size = 2664
	#print(np_X.shape)
	#down_arr = down_data_sample(np_X, sample_size)
	#print(down_arr.shape)

	#(x_monstar, y_monstar) = patch_nparrays(prefix)
	#print(x_monstar.shape)
	#print(y_monstar.shape)
	#arr = "Y_train.npy"
	#npy2text(arr, 'np2txt.txt')
	