#/usr/bin/env/python
# created by nebulaM78 team; capstone 2020;
# to create and update a dictionary that maps the sign to an index in order to generate
# its corresponding training index-label;

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

# implementation;
#   0. we have a dictionary containing the pair of: {"auslan sign", x}
#       where x is its corresponding integer label;
#   1. given a filelist of the signs the team has managed to find;
#       scan the filelist to update any new sign, and generate its
#       corresponding index-label and add it;
#   2. save and load the dictionary;
#   3. generate the integer-label based on the number of entries;

# global constant;
# file format;
FILE_FORMAT = "json"

def update_dict(src_path, saved_path = "", save_name = 'saved_dict.p'):
	'''
	args 
		1. src_path; the txt file for the converted video;
		2. saved_path; where to save for the dictionary;
		3. save_name; the name of the dictionary to save;
	return 
		- dictionary
	function
		- create or update the dictionary; then save;
	'''
	# check whether we have an existing dictionary;
	# src - https://stackoverflow.com/questions/28633555/how-to-handle-filenotfounderror-when-try-except-ioerror-does-not-catch-it
	#to load the saved dictonary;
	dict_path = saved_path + save_name
	 
	try:
	  with open(dict_path, 'rb') as fp:
		   dict = pickle.load(fp)
		   print('the existing dictionary has been loaded\n', dict)
		   dict = check_new_key(dict, src_path)
	except OSError as e:
		 print("e\n Need to create new dictionary\n")
		 dict = create_dict(src_path)
	
	# done everything? save it then;
	with open(dict_path, 'wb') as fp:
		pickle.dump(dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
		print("the dictionary has been saved;\n")

	return dict
		
def create_dict(src_path):
	'''
	args 
		1. src_path; the txt file for the converted video;
	return 
		- the created dictionary
	function
		- create a new dictionary; then save;
	'''
	dict = {}
	# starting index;
	index = 1 # cannot be zero; this has been taken care of;

	# get the class name;
	tmpstring = (src_path.split('\\'))[-1]
	tmpstring = (tmpstring.split('.'))[0]
	classname = (tmpstring.split('_'))[0]
	
	print("new element:\n", classname)
	dict[classname] = index
	
	# save it at the current directory by default;
	# src - https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
	print("the newly created dictionary;\n", dict)

	# done?
	return dict

def check_new_key(dict, src_path):
	'''
	args 
		1. the dictionary	
		2. src_path; the txt file for the converted video;   
	return 
		- the dictioary
	function
		- find new key and insert to the dictonary;
	'''
	# update with the most recent index;
	oldlen = len(dict)
	index = oldlen + 1
	
	# safeguards;
	print("checking for a new key for the dictionary, if exists\n")
	print('asserting that we have a non-empty dictionary here\n')
	assert(oldlen != 0)

	# get the class name;
	tmpstring = (src_path.split('\\'))[-1]
	tmpstring = (tmpstring.split('.'))[0]
	classname = (tmpstring.split('_'))[0]
	
	# no need to "reupdate the dictionary if the key already exists
	if not (classname in dict):
		print("new element found;\n", classname)
		dict[classname] = index
		index += 1
		print("the dictionary has been updated\n")
	else:
		print("no new element has been found;\n")

	# done?
	return dict


# test driver;
if __name__ == '__main__':

	src_path = "C:\\Users\\yongw4\\Desktop\\image-database\\ambulance.txt"
	save_name = 'saved_dict.p'
	dict = update_dict(src_path, saved_path = "", save_name = 'saved_dict.p')
	print(dict)