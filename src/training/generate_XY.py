#/usr/bin/env/python
# to generate X, Y.txt files from a database for training;
# created by nebulaM78 team; capstone 2020;

import file_tools as ftool
import sign2index_dict as dict_m
from pprint import pprint
import glob, os
import sys


def generate_XY(txt_path, path_X,  path_Y):
	'''
	args:
		1. src_path; the txt file for the converted video;
		2. path_X, where to save the X.txt;
		3. path_Y, where to save the Y.txt;
	return:
		none;
	function:
		generate the labels: Y and X for one individual video in txt format;

	'''
	#-----------------------------------------------
	# copy the individual txt to the X.txt;
	#-----------------------------------------------
	with open(path_X, 'a') as fileX, open(txt_path, 'r') as fileVideo:
		print('given txt_path\n', txt_path)
		print("Appending ... \n")
		for line in fileVideo:
			# safe guard;
			assert(line != "")
			#print("txt line, \n", line)
			fileX.write(line)
		print("finished appending\n")

	#-----------------------------------------------
	# generate its corresponding Y label for Y.txt;
	#-----------------------------------------------
	# first, update the dict in case of new entries;
	dict = dict_m.update_dict(txt_path, saved_path = "", save_name = 'saved_dict.p')

	# get the classname;
	[classname, image_checked] = ftool.checkoff_file(txt_path, "checked")
	print('classname\n', classname)
	try:
			ylabel = dict[classname]
			print("the class the image belongs to ", ylabel)
	except KeyError as e:
			print("the key doesnt exit!\n", e)	
			print("system abort\n")
			sys.exit(-1)

	# write the corresponding label to Y.txt
	with open(path_Y, 'a') as fileY:
		# add EOL to conform to txt format;
		insertline = str(ylabel) + "\n"
		fileY.write(insertline)
		print("done labelling\n")

	
# test driver;
if __name__ == '__main__':
	# constant paths;
	image_dataset_path = "C:\\Users\\yongw4\\Desktop\\output_eval_alpha\\alphabet_Z.txt"
	path_X = "C:\\Users\\yongw4\\Desktop\\output_eval_alpha\\X_val.txt"
	#path_Y = "C:\\Users\\yongw4\\Desktop\\Y_train.txt"
	generate_XY(image_dataset_path, path_X, "")