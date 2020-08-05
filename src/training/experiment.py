import cv2
import time
import sys
import numpy as np
import ffmpeg
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import os

# src - https://github.com/Zulko/moviepy/issues/813

#from moviepy.editor import *
PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
input = PREFIX + "ambulance_14.mp4"
output = PREFIX + "output_test_2.mp4"

def get_frame_count_CV(input):
	cap = cv2.VideoCapture(input)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	input_fps = int(cap.get(5))
	total_frames = int(cap.get(7))
	print("CV2; input_fps ", input_fps)
	print("CV2; frame count ", total_frames)
	return total_frames

def get_frame_count_MP(input):
	clip = mp.VideoFileClip(input)
	num_frames = len(list(clip.iter_frames()))
	print("movidepy; num_frames: ", num_frames)
	return num_frames

def lstm_window_check(input_list):
	# assert the length >= 75, otherwise pad it with anything
	# to make up to the required window width;
	window_width = 77 # leave some room: 75 + buffer;
	length = len(input_list)
	if(length < window_width):
		# pad it with the last frame;
		grab = input_list[-1]
		diff = window_width - length
		for i in range(0, diff):
			input_list.append(grab)
	# checked; safe now;
	return input_list

def frames2video(input_list, size):
	# encode it with its bare ascii code; a wraparound;
	# src - https://forums.developer.nvidia.com/t/python-what-is-the-four-characters-fourcc-code-for-mp4-encoding-on-tx2/57701/4
	fourcc = 0x7634706d
	out = cv2.VideoWriter(output,fourcc, 30, size)
	for i in range(len(input_list)):
		# writing to a image array
		out.write(input_list[i])
	out.release()

def video_flip(input, output):
	video = mp.VideoFileClip(input)
	out = video.fx(vfx.mirror_x)
	out.write_videofile(output)
	
def video_rotate(input, output, degree):
	clip = mp.VideoFileClip(input)
	newclip = (clip.fx( vfx.rotate, degree))
	newclip.write_videofile(output)
	print('the rotated video has been saved to : ', output)

def slow_video(input, output, speed):
	clip = mp.VideoFileClip(input) 
	slow_clip = (clip.fx( vfx.speedx, speed))
	# get the (width, height)
	size = slow_clip.size
	input_list = (list(slow_clip.iter_frames()))
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
	frames2video(extract, size)
	print("the slowed-video has been saved to: ", output)

def fast_video(input, output, speed):
	clip = mp.VideoFileClip(input) 
	fast_clip = (clip.fx( vfx.speedx, speed))
	# get the (width, height)
	
	size = fast_clip.size
	input_list = (list(fast_clip.iter_frames()))
	# assert we meet the lstm window width minimum;
	
	input_list = lstm_window_check(input_list)
	# done? write it;
	frames2video(input_list, size)
	print("the fast-video has been saved to: ", output)

def video_speed(input, output, speed):
	speed = abs(speed)
	if (speed <= 1):
		slow_video(input, output, speed)
	else:
		fast_video(input, output, speed)

if __name__ == '__main__':
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	input = PREFIX + "ambulance_1.mp4"
	output = PREFIX + "output_test_3.mp4"
	input1 = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_1.mp4"
	clip = mp.VideoFileClip(input)
	ls = list(clip.iter_frames())
	#print(ls[len(ls)])
	get_frame_count_CV(input)
	#speed = 1.3, 1.5
	#fast_video(input, output, speed = 1.3)
	get_frame_count_CV(output)
	
	#speed = 0.5, 0.6, 0.8
	slow_video(input, output, speed = 0.6)
	video_rotate(input,output , degree = -7)
	#get_frame_count_CV(output)
	flipped_output = PREFIX + "output_test_4.mp4"
	
	video_flip(output, flipped_output)
	
		#return (input_list[int(middle)], input_list[int(middle-1)])
	
	
	#num_frames = len(list(clip.iter_frames()))
	#video_zoom()
	#video_speed()
	'''
	speed = 0.7 # 0.8
	speed = 0.6
	video_speed(input, output, speed, 0, 0)
	get_fps_CV(input)
	get_fps_MP(input)
	get_fps_CV(output)
	get_fps_MP(output)
	'''