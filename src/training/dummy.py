import sys
import LSTM_tools as lstm
import numpy as np
import os


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
				dataset = lstm.load_X(src_path)
			# the Y label;
			else:
				dataset = lstm.load_Y(src_path)
			# done selecting the files?
			PATCH[index].append(dataset)

	# concatenate all the arrays into one;
	X_monstar = np.concatenate(tuple(PATCH[0]), axis = 0)
	Y_monstar = np.concatenate(tuple(PATCH[1]), axis = 0)
	return (X_monstar, Y_monstar)

# test driver;
if __name__ == '__main__':
	txt_directory = "C:\\Users\\yongw4\\Desktop\\FATE\\txt-files"
	for root, dirs, files in os.walk(txt_directory, topdown=False):
		print('roots: ', root)
		print('dirs: ', dirs)
		print('files: ', files)

			
	#X_monstar, Y_monstar = patch_nparrays(txt_directory)
	#print(X_monstar.shape)
	#print(Y_monstar.shape)
	