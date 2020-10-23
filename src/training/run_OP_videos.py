import synthetic_tools as syntools
import os
import file_tools as ftools
import sys
import video_tools as VID
import frame_sampling as FRAME

# global var(s);
#SPEED = [1, 0.6, 0.8, 1.2, 1.4]

SPEED = [1.0]
#DEGREE = [0, 3, 5, 7, 9, -3, -5, -7, -9]


def process_one_video(input, path_X, path_Y, func, DEGREE, SPEED):
	for i in range(len(SPEED)):
		speed_seed = SPEED[i]
		syntools.synthesize_block(input, speed_seed, path_X, path_Y, func, DEGREE)
		#syntools.synthesize_block(input, speed_seed, path_X, path_Y)

def process_one_block(signvideodirectory, path_X, path_Y, func, parameters=[0], speed = SPEED):
	
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


	# which transformatoon? rotation or shear?
	#func = VID.shear_video
	func = VID.video_rotate
	# transformation parameters; angles?
	#DEGREE = [-10, -5, 5, 10]
	#DEGREE = [0, 3, 5, 7, 9, -3, -5, -7, -9]

	for root, dirs, files in os.walk(signvideodirectory, topdown=False):
		for name in files:
			# make sure the current file is MP4;
			ext = name.split('.')[-1]
			if (ext != "mp4"):
				continue
			# OK, it's mp4; process it;
			src_path = os.path.join(root, name)
			print("currently processing: ", src_path)
			#sys.exit('DEBUG')
			# current video has not been processed; 
			if not (ftools.checksubstring(src_path, "checked")):
				process_one_video(src_path, path_X, path_Y, func, parameters, speed)
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
	
def reset_direc_filename(path):
	'''
	 a quick fix to reset the "checked-off" files in a directory
	'''
	for root, dirs, files in os.walk(path, topdown=False):
		for name in files:
			# make sure the current file is MP4;
			ext = name.split('.')[-1]
			if (ext != "mp4"):
				continue
			
			# OK, it's mp4; process it;
			src_path = os.path.join(root, name)
			#print("currently processing: ", src_path)

			# remove the checkoff tag;
			if(("checked" in name)):
				tmp = name.split("_")
				classname = tmp[0]
				iter = tmp[1]
				reset =  classname + "_" + iter + ".mp4"
				tmppath =  os.path.join(root, reset)
				os.rename(src_path, tmppath)
			

def drive_test_pipeline(test_path):
	'''
	1. apply completely new video transformations to synthesize more test data;
	2. generate ONE txt file;
	3. convert the txt into np (75-frames)
	4. down-sampling it into 35-frames by including random shuffling;
	5. output the final np array and save it;
	'''
	# get the class name to name the txt files accordingly;
	tmpname = test_path.split("\\")[-1]

	# transformation set to synthesize the test data;
	henshin = [VID.warp_phi_video, VID.warp_theta_video, VID.zoom_video]
	
	# its corresponding parameters;
	params = [[-40,-20,0,20,40], [-40,-20,20,40], [1.4, 1.8]]
	
	# write paths;
	path_X = os.path.join(test_path, "X_" + tmpname + "_test.txt")
	path_Y = os.path.join(test_path, "Y_" + tmpname + "_test.txt")

	for i in range(0, len(henshin)):
		#process_one_block(test_path, path_X, path_Y, func = henshin[i], PARAMS[i])
		process_one_block(test_path, path_X, path_Y, henshin[i], params[i], speed = [1])
		# removing the "checked-off" tag by resetting the directory;
		reset_direc_filename(test_path)
   
	# convert the generated txt files into 35 - type and save it;
	FRAME.gen_XY(test_path)
	

		
if __name__ == '__main__':
	path = "C:\\Users\\yongw4\\Desktop\\test\\HOSPITAL\\yick"
	path = "C:\\Users\\yongw4\\Desktop\\test-set\\test-set\\organized"
	#path = "C:\\Users\\yongw4\\Desktop\\test\\HOSPITAL\\yick"
	drive_test_pipeline(path)
	#drive_test_pipeline(path)
	#process_one_block(path)
	'''
	for root, dirs, files in os.walk(path):
		for i in range(len(dirs)):
			classvideo = os.path.join(root, dirs[i])
			print("processing class: ", classvideo)
			process_one_class(classvideo)
		# stop here;
		break
	'''