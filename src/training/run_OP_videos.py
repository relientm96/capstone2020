import synthetic_tools as syntools
import os
# global var(s);
SPEED = [1, 0.6, 0.8, 1.2, 1.4]
def process_one_video(input, path_X, path_Y):
	for i in range(len(SPEED)):
		speed_seed = SPEED[i]
		syntools.synthesize_block(input, speed_seed, path_X, path_Y)

if __name__ == '__main__':
	path_X = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\X_dummy.txt"
	path_Y = "C:\\Users\\yongw4\\Desktop\\test_synthesis\\Y_dummy.txt"

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
	input = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_14.mp4"
	process_one_video(input, path_X, path_Y)

