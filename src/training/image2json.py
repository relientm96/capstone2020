#/usr/bin/env/python
# to convert all the images of one class to json files
# created by tsz kiu; 

import preprocessing as pp # tsz kiu's class
import json
from pprint import pprint
import glob, os
import sys
#====================== implementation =======================
# code structure;
#   - open the folder which corresponds to one class;
#   - in it, we have all the images of the same class;
#   - run:
#       - openpose on all the images;
#       - output json files;

# file/folder name structure;
# assumptions;
#   - each folder is named based on the class; say "A"
#   - each image has been indexed;
#   - for example: folder: "A" --> files: {"a01.jpg", ..., "a100.jpg"}

# output structure;
#   for each .jpg, we have its corresponding .json file
#   for each class, group all the converted json file into one folder;
#   example: 
#       folder name: "A-json" --> "a01.json, ... , a100.json"
#============================================================


def image2json(imagfolder_path, outputfolder_path):
	'''
	args:
		1. "imagfolder_path"; the folder containing the set of images of one class;
		2. "outputfolder_path", the corresponding folder containing all the converted json files;
			- such folder may not necessarily exist;
	return:
		nothing;
	function:
		run openpose on images and output json;
	'''
	#----------------------------------------------------------------------
	# directories handling;
	#----------------------------------------------------------------------
	# imagfolder_path;
	if not os.path.exists(imagfolder_path):
		print("the source file path is not valid;\n")
		sys.exit(0)
	# outputfolder_path;
	if not os.path.isdir(outputfolder_path):
		try:
			print("creating a directory for the json files;\n")
			os.mkdir(write_path)
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)
	else:
		print('the directory to store the json files has been created\n')

	# append the regex "*.jpg" to the folder path
	imagfolder_path = imagfolder_path + "*.jpg"
	#----------------------------------------------------------------------
	# real work starts here;
	#----------------------------------------------------------------------
	# Create OpenPoseProcessor Object
	opProc =  pp.OpenPoseProcessor()
	for imagepath in sorted(glob.glob(os.path.join(imagfolder_path))):
		print(imagepath)
		opProc.run_openpose(imagepath, outputfolder_path)

# test driver;
if __name__ == '__main__':
	src_path = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-images\\"
	write_path = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-json\\"
	image2json(src_path, write_path)