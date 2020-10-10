#/usr/bin/env/python
# to generate X, Y.txt files from a database for training;
# created by nebulaM78 team; capstone 2020;

import file_tools as ftool
import sign2index_dict as dict_m
from pprint import pprint
import glob, os
import sys

# generate the label, Y and X for one individual image
def process_XY(image_path, image_dataset_path,  path_X, path_Y):
	#-----------------------------------------------
	# copy the individual txt to the X.txt;
	#-----------------------------------------------
	with open(path_X, 'a') as fileX, open(image_path, 'r') as fileImage:
		print("Appending ... \n")
		for line in fileImage:
			#print(line)
			fileX.write(line)
		print("finished appending\n")

	#-----------------------------------------------
	# generate its corresponding Y label for Y.txt;
	#-----------------------------------------------
	# first, update the dict in case of new entries;
	dict = dict_m.update_dict(image_dataset_path, saved_path = "", save_name = 'saved_dict.p')
	# get the classname;
	[classname, image_checked] = ftool.checkoff_file(image_path, "checked")
	print('classname\n', classname)
	try:
			ylabel = dict[classname]
			print("the class the image belongs to \n", ylabel)
	except KeyError as e:
			print("the key doesnt exit!\n", e)	
			print("system abort\n")
			sys.exit(-1)

	# write the corresponding label to Y.txt
	with open(path_Y, 'a') as fileY:
		# add EOL to conform to txt format;
		insertline = str(ylabel) + "\n"
		fileY.write(insertline)
		print("done labelling for one image\n")

	#-----------------------------------------------
	# rename the relevant filee for checkoff;
	#-----------------------------------------------
	os.rename(image_path, image_checked)
	# # done for one image; continue;

# generate X and Y on multiple (non-repetitive) images;
def generate_XY(image_dataset_path, path_X, path_Y):
    # append the regex;
    search_path = image_dataset_path + "*.txt"
    for image_path in sorted(glob.glob(os.path.join(search_path))):
        has_checked = ftool.checksubstring(image_path, "checked")
        if not has_checked:
            process_XY(image_path, image_dataset_path, path_X, path_Y)
        else:
            print("current image has been checked off; continue\n")
    # done iterating;
    print("all entries have been checked off\n")
      

# test driver;
if __name__ == '__main__':
    # constant paths;
    #image_dataset_path = "C:\\Users\\yongw4\\Desktop\\train-image-database\\"
    image_dataset_path = "C:\\Users\\yongw4\\Desktop\\output_eval_alpha\\alphabet_Z.txt"
    path_X = "C:\\Users\\yongw4\\Desktop\\output_eval_alpha\\X_val.txt"
    #path_Y = "C:\\Users\\yongw4\\Desktop\\Y_train.txt"
    generate_XY(image_dataset_path, path_X, "")