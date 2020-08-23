import synthetic_tools as syntools
import os
import file_tools as ftools
import sys

# global var(s);
#SPEED = [1, 0.6, 0.8, 1.2, 1.4]

# date - 24 aug, 2020;
SPEED = [1]
def process_one_video(input, path_X, path_Y):
	for i in range(len(SPEED)):
		speed_seed = SPEED[i]
		syntools.synthesize_block(input, speed_seed, path_X, path_Y)
	

if __name__ == '__main__':
	signvideodirectory = "C:\\Users\\yongw4\\Desktop\\FATE"
	
	path_X = os.path.join(signvideodirectory, "X_train.txt")
	path_Y = os.path.join(signvideodirectory, "Y_train.txt")
	
	# safeguard;
	# create them if the files do not exist;
	print("Checking if the X.txt and Y.txt exist ...\n")
	if not (os.path.exists(path_X) or os.path.exists(path_Y)):
		try:
			open(path_X, 'w').close()
			open(path_Y, 'w').close()
		except Exception as e:
			print("An error occured", e)
			sys.exit(-1)

	for root, dirs, files in os.walk(signvideodirectory, topdown=False):
		for name in files:
			# make sure the current file is MP4;
			ext = name.split('.')[-1]
			if (ext != "mp4"):
				continue
			# OK, it's mp4; process it;
			src_path = os.path.join(root, name)
			print("currently processing: ", src_path)
			# current video has not been processed; 
			if not (ftools.checksubstring(src_path, "checked")):
				process_one_video(src_path, path_X, path_Y)

				# done processing? sign off;
				# so that the processed video will not be processed again;
				print('the current video has been processed: ', src_path)
				[_, dst_checked] = ftools.checkoff_file(src_path, "checked")
				os.rename(src_path, dst_checked)
			# have processed;
			else:       
				print("the current video has already been processed, skip\n")

            # debugging ...
			#sys.exit("stop at one-video to check;")
		