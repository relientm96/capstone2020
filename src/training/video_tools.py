#/usr/bin/env/python
# to start window to record video;
# adapted and modified by nebulaM78 team; capstone 2020;
import cv2
import time
import sys
import tensorflow as tf
import tempfile
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os
import imageio
from matplotlib import image
import matplotlib.pyplot as plt

# needed to save non-string python object; eg dictionary;
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

#------------------------------------------------------------------------------------------
# OVERVIEW: 
# it has two set of modules:
# 1. tools for sanity checking;
# 2. tools to transform videos;
#
# ASSUMPTIONS;
# 1. only mp4 format;
# 2. the window width for lstm is fixed at 75-size;
# 3. by (2), the transformed video's frame count will have at least size of 75;
#------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
# TOOLS FOR SANITY CHECK ON OPENPOSE VIDEOS AND AUSLAN RECOGNITION:
# 1. record_video(); this is our offline auslan prediction;
# 2. annotate_video(); to insert text onto the video;
# *acknowledgement;
# src = https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# src - https://raspberrypi.stackexchange.com/questions/66976/capture-video-for-a-certain-time-then-quit-and-save-to-a-folder-using-opencv-3
# src - https://www.geeksforgeeks.org/python-opencv-write-text-on-video/
# src - https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
#------------------------------------------------------------------------------------------

def record_video(filename, signtime = 3, preptime = 3):
	'''
	args:
		- filename; where to save the recorded video?
		- signtime; how long do you want to capture the actual sign;
		- preptime; how long do you need to prepare before signing;
	return:
		- None;
	function:
		- start up a self recording window and save it;
	'''
	# capture frames from a camera with device index=0
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	cap = cv2.VideoCapture(0)
	fps = 20    # h0ow many frames you want to write in a second;
	print('writing fps: ', fps)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

	# Define the duration (in seconds) of the video capture here
	#preptime = 3
	buffertime = 0.3
	#signtime = 3
	capture_duration = preptime + signtime + buffertime
	start_time = time.time()
		
	while( int(time.time() - start_time) < capture_duration ):
		# reads frame from a camera 
		_, frame = cap.read() 
	
		# describe the type of font 
		# to be used. 
		font = cv2.FONT_HERSHEY_SIMPLEX 
  
		# give a head up to the user before the actual signing;
		# no need to save the frame within this interval;
		tick = int(time.time() - start_time)
		if(tick <= preptime):
			TEXT = "Start signing in " + str(tick)
			cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4)
			cv2.imshow('Camera',frame) 

		# the user should be prepared now;
		else:
			TEXT = "OOO, Handsome! " + "tick: " + str(tick-preptime)
			cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4) 
 
			# Display the frame
			cv2.imshow('Camera',frame) 
			# write the frames here to the file;
			out.write(frame)
			# user could force shut down by pressing "Q"
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		end_recordtime = time.time()

	# sanity check on the total time taken;
	print('sanity check:\n total time taken: ', end_recordtime - start_time)
	print('whole capture duration: ', capture_duration)
	
	# release the camera from video capture
	cap.release() 
	
	# De-allocate any associated memory usage 
	cv2.destroyAllWindows() 

	print('the video has been saved to:', filename)

def annotate_video(filename, info_list):
	'''
	args:
		- filename; where to load the saved video?
	return:
		- None;
	function:
		- edit the saved video but will not save the edited version;
		- this serves as auxiliary tool for sanity checking;
	'''
	cap = cv2.VideoCapture(filename)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
		sys.exit("cv2 cannot open the video stream")
	input_fps = int(cap.get(5))
	total_frames = int(cap.get(7))

	print('input fps: ', input_fps)
	print('number of frames in the input video: ', total_frames)
	stepsize = 15
	future_index = 0

	count_frame = 1
	track = 0
	while(cap.isOpened()):
		# prevent out-of-range error;
		if not (future_index > (len(info_list)-1)):
			TEXT = str(info_list[future_index])
			font = cv2.FONT_HERSHEY_SIMPLEX 

		# update the sign every 0.5 second (15 frames);
		if((count_frame % stepsize) == 0):
			future_index += 1
			track += 1
		# process the frame;
		ret, frame = cap.read()
		#print(np.shape(frame))
		cv2.putText(frame, TEXT, (50, 400),  font, 1,  (0, 255, 255), 2, cv2.LINE_4) 
 
		cv2.imshow('frame', frame)
		# set it to "30" to slow down the playback;
		# note: the higher this value is, the slower the playback is;
		if cv2.waitKey(input_fps*2) & 0xFF == ord('q'):
			break
		count_frame += 1
		#print('count_frame: ', count_frame)
		# after all the frames have been read, no point to continue;
		if(count_frame > total_frames-1):
			break
	cap.release()
	cv2.destroyAllWindows()
	print('track: ', track)


#------------------------------------------------------------------------------------------
# VIDEO TRANSFORMATIONS;
# this set contains the tools to transform the videos and some safeguards;
# 1. get_frame_count;
# 2. lstm window check; to check whether the minimum window width has been met;
# 3. horizontal flipped;
# 4. change film speed while maintaining the frame count;
# 5. rotation;
# 6. convert video to frames;
# 7. combine frames and save it as a video;
# *acknowledgement;
# src - https://zulko.github.io/moviepy/index.html
#------------------------------------------------------------------------------------------

def get_frame_count_CV(input):
	'''
	function:
		- to get the frame count using CV2 library;
	args:
		- the complete mp4 filename (incl, path);
	return: 
		- the total number of frames in the video;
	'''
	cap = cv2.VideoCapture(input)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	input_fps = int(cap.get(5))
	total_frames = int(cap.get(7))
	print("CV2; input_fps ", input_fps)
	print("CV2; frame count ", total_frames)
	cap.release()
	cv2.destroyAllWindows()
	return total_frames

def get_frame_count_MP(input):
	'''
	function:
		- to get the frame count using moviepy library;
	args:
		- the complete mp4 filename (incl, path);
	return: 
		- the total number of frames in the video;
	'''
	clip = mp.VideoFileClip(input)
	num_frames = len(list(clip.iter_frames()))
	print("movidepy; num_frames: ", num_frames)
	
	del clip.reader
	del clip

	return num_frames

def lstm_window_check(input_list):
	'''
	function:
		- assert the length >= 75, otherwise pad it with anything
		- to make up to the required window width;
	args:
		- a list of the video frames;
	return:
		- the input_list with the "correct" length;
	'''
	window_width = 77 # leave some room: 75 + buffer;
	length = len(input_list)
	if(length < window_width):
		# pad it with a black frame;
		grab = (input_list[-1])
		grab.fill(0)
		diff = window_width - length
		for i in range(0, diff):
			input_list.append(grab)
	# checked; safe now;
	return input_list

def grab_frames(input):
	'''
		function:
			- use opencv to convert a video to frames in list;
		args:
			- input file;
		return:
			- a list of the frames;
	'''
	cap = cv2.VideoCapture(input)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
		sys.exit("cv2 cannot open the video stream")
	# get some metadata;
	input_fps = int(cap.get(5))
	total_frames = int(cap.get(7))
	
	count_frame = 0
	store = []
	while(cap.isOpened()):
		# process the frame;
		ret, frame = cap.read()
		count_frame += 1
		#cv2.imshow('frame', frame)
		store.append(frame)
		#if cv2.waitKey(30) & 0xFF == ord('q'):
		#	break
		# after all the frames have been read, no point to continue;
		# a wrap around to prevent opencv2 error
		if(count_frame > total_frames-1):
			break
	cap.release()
	cv2.destroyAllWindows()
	return store	




def frames2video(input_list, output, size):
	'''
	function:
		- to convert a list of frames to a video;
	args:
		- input_list; the list of frames;
		- size; the (width, height) of the frame;
	return;
		- none;
	'''
	# encode it with its bare ascii code; a wraparound;
	# src - https://forums.developer.nvidia.com/t/python-what-is-the-four-characters-fourcc-code-for-mp4-encoding-on-tx2/57701/4
	fourcc = 0x7634706d
	out = cv2.VideoWriter(output,fourcc, 30, size)
	for i in range(len(input_list)):
		# writing to a image array
		out.write(input_list[i])
	out.release()
	cv2.destroyAllWindows()
	

def video_flip(input, output):
	'''
	function:
		- to horizontally flip the video and save as a new video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
	return:
		- none;
	'''
	video = mp.VideoFileClip(input)
	out = video.fx(vfx.mirror_x)
	out.write_videofile(output)
	print("the flipped video has been saved to: ", output)
	del video.reader
	del video

	
def video_rotate(input, output, degree):
	'''
	function:
		- to rotate a video by ?? degree, and save as a new video
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- degree; clockwise if positive; otherwise; anticlockwise;
	return:
		- none;
	'''

	clip = mp.VideoFileClip(input)
	newclip = (clip.fx( vfx.rotate, degree))
	newclip.write_videofile(output)
	print('the rotated video has been saved to : ', output)
	del clip.reader
	del clip


def slow_video(input, output, speed):
	'''
	function:
		- to slow down a film while maintaining the frame count, 
		- and save it as a new video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- speed; must be <= 1;
	return:
		- none;
	'''
	
	clip = mp.VideoFileClip(input) 
	slow_clip = (clip.fx( vfx.speedx, speed))
	# get the (width, height)
	size = slow_clip.size
	with tempfile.TemporaryDirectory() as dummy_path:
		dummy = dummy_path + "\\dummy.mp4"
		slow_clip.write_videofile(dummy)
		input_list = grab_frames(dummy)
	length = len(input_list)
	# assert we meet the lstm window width minimum;
	input_list = lstm_window_check(input_list)
	# now, it's safe to extract the centre frame ...
	# of size 75;
	middle = float(len(input_list))/2
	hardcode =37
	# consider odd and even list length separately;
	if(middle % 2 != 0):
		middle = int(middle - .5)
		extract = input_list[middle-hardcode:middle+1+hardcode]
	else:
		middle = int(middle-1)
		extract = input_list[middle-(hardcode-1):middle+2+hardcode]
	# done? write it;
	frames2video(extract, output, size)
	print("the slowed-video has been saved to: ", output)
	# done? clean up;
	# src - https://github.com/Zulko/moviepy/issues/57
	del clip.reader
	del clip

def fast_video(input, output, speed):
	'''
	function:
		- to speed up a film while maintaining the frame count, 
		- and save it as a new video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- speed; must be > 1;
	return:
		- none;
	'''
	clip = mp.VideoFileClip(input) 
	fast_clip = (clip.fx( vfx.speedx, speed))
	# get the (width, height)
	size = fast_clip.size
	with tempfile.TemporaryDirectory() as dummy_path:
		dummy = dummy_path + "\\dummy.mp4"
		fast_clip.write_videofile(dummy)
		input_list = grab_frames(dummy)
	
	# assert we meet the lstm window width minimum;
	input_list = lstm_window_check(input_list)
	# done? write it;
	frames2video(input_list, output, size)
	print("the fast-video has been saved to: ", output)
	# clean up;
	# src - https://github.com/Zulko/moviepy/issues/57
	del clip.reader
	del clip



def video_speed(input, output, speed):
	'''
	function:
		- to change the speed of  a film while maintaining the frame count, 
		- and save it as a new video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- speed; slower if < 1; faster if > 1; otherwise, unchanged;
	return:
		- none;
	'''
	if (speed <= 1):
		slow_video(input, output, speed)
	else:
		fast_video(input, output, speed)	


def shear_video(input, output, shear=-10):
	# ref - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_affine_transform
	'''
	function:
		- to shear  video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- shear angle;
	return:
		- none;
	'''
	# get size;
	clip = mp.VideoFileClip(input) 
	size = tuple(clip.size)
	del clip.reader
	del clip
	
	input_list = grab_frames(input)
	# assert we meet the lstm window width minimum;
	input_list = lstm_window_check(input_list)
	outls = []
	# apply the transformation:
	for i in range(0, len(input_list)):
		frame = input_list[i]
		henshin = tf.keras.preprocessing.image.apply_affine_transform(frame, theta=0, tx=0, ty=0, shear=shear, zx=1.1, zy=1.1,
																		row_axis=0, col_axis=0, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
		
		#plt.imshow(henshin)
        #plt.show()
        #henshin = cv2.cvtColor(henshin, cv2.COLOR_RGB2BGR)
		outls.append(henshin)
		
	# done? write it;
	frames2video(outls, output, size)
	
	#video_rotate(output, "viola.mp4", -10)
	print("the sheared-video has been saved to: ", output)


def zoom_video(input, output, ratio = 1.2):
	# ref - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_affine_transform
	'''
	function:
		- to zoom out a video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- ratio;
	return:
		- none;
	'''
	# get size;
	clip = mp.VideoFileClip(input) 
	size = tuple(clip.size)
	del clip.reader
	del clip
	
	input_list = grab_frames(input)
	# assert we meet the lstm window width minimum;
	input_list = lstm_window_check(input_list)
	outls = []
	# apply the transformation:
	for i in range(0, len(input_list)):
		frame = input_list[i]
		henshin = tf.keras.preprocessing.image.apply_affine_transform(frame, theta=0, tx=0, ty=0, shear=0, zx=ratio, zy=ratio,
																		row_axis=0, col_axis=0, channel_axis=2, fill_mode='constant', cval=0.0, order=1)
		
		#plt.imshow(henshin)
        #plt.show()
        #henshin = cv2.cvtColor(henshin, cv2.COLOR_RGB2BGR)
		outls.append(henshin)
		
	# done? write it;
	frames2video(outls, output, size)
	
	#video_rotate(output, "viola.mp4", -10)
	print("the zoomed-out-video has been saved to: ", output)

def warp_video(input, output, shear=-10):
	# ref - https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_affine_transform
	'''
	function:
		- to shear  video;
	args:
		- input; the filename (path) of the input video;
		- output; where to save?
		- shear angle;
	return:
		- none;
	'''
	# get size;
	clip = mp.VideoFileClip(input) 
	size = tuple(clip.size)
	del clip.reader
	del clip
	
	input_list = grab_frames(input)
	# assert we meet the lstm window width minimum;
	input_list = lstm_window_check(input_list)
	outls = []
	# apply the transformation:
	for i in range(0, len(input_list)):
		frame = input_list[i]
		henshin = tf.keras.preprocessing.image.apply_affine_transform(frame, theta=0, tx=0, ty=0, shear=shear, zx=1.1, zy=1.1,
																		row_axis=0, col_axis=0, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
		
		#plt.imshow(henshin)
        #plt.show()
        #henshin = cv2.cvtColor(henshin, cv2.COLOR_RGB2BGR)
		outls.append(henshin)
		
	# done? write it;
	frames2video(outls, output, size)
	
	#video_rotate(output, "viola.mp4", -10)
	print("the sheared-video has been saved to: ", output)
	
# test driver;
if __name__ == '__main__':

	# shearing;
	videopath = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-videos\\auslan\\ambulance.mp4"
	output = "C:\\CAPSTONE\\capstone2020\\src\\training\\test-videos\\auslan\\shear_ambulance.mp4"
	videopath = "C:\\Users\\yongw4\\Desktop\\dummy_video.mp4"
	output = "C:\\Users\\yongw4\\Desktop\\henshin_video.mp4"
	zoom_video(videopath, output, 1.5)

	#video_speed(videopath, output, 0.6)
	# test - 01
	'''
	filename = "C:\\Users\\yongw4\\Desktop\\JSON\\" + 'dummy.avi'
	fps = 30.0
	record_video(filename,  signtime = 3)
	'''

	# test - 02
	'''
	list_path = "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\"+ 'future_prediction.txt'
	with open(list_path, "rb") as fp:   
		info_list = pickle.load(fp)
	filename = "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\" +  "result.avi"
	annotate_video(filename, info_list)
	'''
	'''
	# test 03;
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	input = PREFIX + "help_95.mp4"
	output = PREFIX + "output_test.mp4"
	video_rotate(input, output, 90)
	'''
	
	'''
	# test 04
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	input = PREFIX + "fullbody.mp4"
	output = PREFIX + "output_test.mp4"
	slow_video(input, output, 0.8)
	'''
	
	'''
	# test 05;
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	input = PREFIX + "help_95.mp4"
	output = PREFIX + "output_test.mp4"
	video2frames(input)
	'''

	