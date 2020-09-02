#/usr/bin/env/python
# auxiliary tool set for file checking purposes;
# created by nebulaM78 team; capstone 2020;

import numpy as np
from pprint import pprint
import glob, os
# needed to save non-string python object; eg dictionary;
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle


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

# combine all the np arrays;
def patch_nparrays(txt_directory):
	'''
		task:
			load all the generated txt files as np arrays and combine into one;
		args:
			txt_directory; the directory where all the txt files are stored;
		returns:
			a tuple of (X, Y)
	'''
	PATCH = [[],[]]
	for root, dirs, files in os.walk(txt_directory, topdown=False):
		for index, name in enumerate(files):
			src_path = os.path.join(root, name)
			# the X file;
			if(index == 0):
				dataset = load_X(src_path)
			# the Y label;
			else:
				dataset = load_Y(src_path)
			# done selecting the files?
			PATCH[index].append(dataset)

	# concatenate all the arrays into one;
	X_monstar = np.concatenate(tuple(PATCH[0]), axis = 0)
	Y_monstar = np.concatenate(tuple(PATCH[1]), axis = 0)
	return (X_monstar, Y_monstar)

# save and load the huge numpy array;
def npy_write(data, filename):
    np.save(filename, data)

def npy_read(filename):
    return np.load(filename)

# test driver;
if __name__ == '__main__':
	
    # patch_nparrays;
	txt_directory = "C:\\Users\\yongw4\\Desktop\\FATE\\speed-01"
	#X_monstar, Y_monstar = patch_nparrays(txt_directory)
	#print(X_monstar.shape)
	#print(Y_monstar.shape)
	