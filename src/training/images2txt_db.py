#/usr/bin/env/python
# to create a database of all the reformatted images in txt;
# created by nebulaM78 team; capstone 2020;


# structure;
#   1. convert all the images to json
#   2. json to txt format
#   3. compile all the txt files;
#   * while maintaining the right label;


import images2json as i2j
import json_image2txt as ji2t
import tempfile # needed for temporary files/directory;
from pprint import pprint
import glob, os


	
# assume this image format to be fixed throughout; 
# shall handle other formats eventually
imageformat = "png"
	
def images2txt_db(imagfolder_path, db_path, imageformat):
	'''
	args:
		1. imagfolder_path; the folder with all the raw images; e.g.
			#   folder: alphabet-1
			#       images: 
			#          -> A1.png
			#           ...
			#          -> Z1.png
		2. db_path; save all the converted image_txt to ...?
	return:
		none;
	function:
		1. # to create a database of all the reformatted images in txt;
	'''
	
	# create a temporary directory to store all the json files;
	# once all images have been processed; 
	# convert these json to txt and save it to the final destionation
	with tempfile.TemporaryDirectory() as dummy_direc:
		#print(dummy_direc)
		i2j.images2json(imagfolder_path, dummy_direc, imageformat)

		# append the regex
		dummy_direc = dummy_direc + "\\*.json"
		for imagepath in sorted(glob.glob(os.path.join(dummy_direc))):
			print(imagepath)
			# get the class name and append it to the txt name so that
			# it's easier to generate the labelling;
			tmpstring = (imagepath.split('\\'))[-1]
			tmpstring = (tmpstring.split('.'))[0]
			txtname = (tmpstring.split('_'))[0]
			save_path = db_path + txtname + '.txt'
			print(save_path)
			ji2t.json_image2txt(imagepath, save_path, 32)
			print("individual txt is being created and saved;\n")
		print("all json images of the same class have been saved as txt")
		
	
# test driver;
if __name__ == '__main__':
	N = 5
	# fixed location for all converted images;
	write_path = "C:\\Users\\yongw4\\Desktop\\image-database\\"
	# run through all alphabet folders;
	for i in range(1, N):
		tmp = "C:\\Users\\yongw4\\Desktop\\alphabets\\alphabets-{CHANGE}\\"
		imageformat = "png"
		src_path = tmp.format(CHANGE = i)
		images2txt_db(src_path, write_path, "png")