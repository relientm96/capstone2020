import cv2
import time
import sys
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
# src - https://timber.io/blog/multiprocessing-vs-multithreading-in-python-what-you-need-to-know/
# src - https://stackoverflow.com/questions/20886565/using-multiprocessing-process-with-a-maximum-number-of-simultaneous-processes
import file_tools as ftools

import multiprocessing as MP
import time
import video_tools as REC
import multi_check as MC

def NANI(input, output, param, jsonpath):
	REC.video_rotate(input, output, param)
	MC.MAIN(output, jsonpath)


if __name__ == '__main__':
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	#input = PREFIX + "ambulance_14.mp4"
	output = PREFIX + "output_test_3.mp4"
	INPUT = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_14.mp4"
	jsonpath = "C:\\Users\\yongw4\\Desktop\\DUMMY_JSON2"
	
	# experiment 1
	# using Processes
	# observation: one process execution at a time with join() method to avoid spamming the gpu which will crash the comp;
	# observation: it took ~128.81336 seconds to openpose process 6 videos of 3-second duration;
	'''
	start_time = time.time()
	#proc = [[video_path, write_path], [video_path2, write_path2]]
	DEGREE = [3,5,7,-3,-5,-7]
	index = 0
	subprocess = []
	for index in range(6):
		## right here
		[classname, _] = ftools.checkoff_file(INPUT, "")
		OUTPUT = PREFIX + classname + "_" + "transformed_" + str(index) + ".mp4"
		p = MP.Process(target = NANI, args=(INPUT, OUTPUT, DEGREE[index], jsonpath))
		# only execute one process at a time;
		# our gpu cannot handle the load!!;
		p.start()
		p.join()
		#index += 1
	#	subprocess.append(p)
	#for pp in subprocess:
	#	pp.join()
	print("--- %s seconds ---" % (time.time() - start_time))
	'''

	# experiment 2
	# using Pool
	# src - https://stackoverflow.com/questions/23816546/how-many-processes-should-i-run-in-parallel
	# src - https://stackoverflow.com/questions/11996632/multiprocessing-in-python-while-limiting-the-number-of-running-processes
	# src - https://stackoverflow.com/questions/868568/what-do-the-terms-cpu-bound-and-i-o-bound-mean
	# observation: two processes execution at a time with Pool method; gpu crashes for > 2 processes!
	# observation: it took ~78.314712 seconds to openpose process 6 videos of 3-second duration;

	start_time = time.time()
	
	num_workers = MP.cpu_count()  
	DEGREE = [3,5,7,-3,-5,-7, 90]
	pool = MP.Pool(3)
	for index in range(len(DEGREE)):
		[classname, _] = ftools.checkoff_file(INPUT, "")
		OUTPUT = PREFIX + classname + "_" + "transformed_" + str(index) + ".mp4"
		pool.apply_async(NANI, args=(INPUT, OUTPUT, DEGREE[index], jsonpath))
	pool.close()
	pool.join()
	print("--- %s seconds ---" % (time.time() - start_time))
	
	