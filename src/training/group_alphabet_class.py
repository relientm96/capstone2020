#/usr/bin/env/python
# tool to rename the files to make alphabets-grouping-by-class convenient;
# created by nebulaM78 team; capstone 2020
from pprint import pprint
import glob, os
import sys

# assumptions;
# 1. the folder contains only one digit;
#   e.g. C:\Capstone\capstone2020\trainingdata\alphabets-01
# 2. only positive integers;

def group_alphabets_class(folder_path, imageformat):

	#----------------------------------------------------------------------
	# directories handling;
	#----------------------------------------------------------------------
	# imagfolder_path;
	if not os.path.exists(folder_path):
		print("the source file path is not valid;\n")
		sys.exit(0)
	# outputfolder_path;
	
	# extract numbers from the given path string;
	# src - https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python/4289415#4289415
	newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in folder_path)
	INDEX = [int(i) for i in newstr.split()][-1]
	tmp  = folder_path + "*." + imageformat
	for imagepath in sorted(glob.glob(os.path.join(tmp))):
		#print(imagepath)
		string = (imagepath.split('.'))[0]
		string02 = (string.split('\\'))
		newname = string02[-1] + str(INDEX) +  '.' + imageformat
		# remove the last element;
		string02.pop()
		string02 = '\\'.join(string02)
		
		dst = string02 + "\\" + newname
		#print(string02)
		#print(dst)
		newfilename = os.rename(imagepath, dst)
		print(newfilename)
	
# test driver;
if __name__ == '__main__':
    N = 7
    for i in range(1, N):
        tmp = "C:\\Users\\yongw4\\Desktop\\alphabets\\alphabets-{CHANGE}\\"
        imageformat = "png"
        src_path = tmp.format(CHANGE = i)
        group_alphabets_class(src_path, imageformat)