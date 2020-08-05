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
	

def video_rotate(input, output, degree):
	clip = mp.VideoFileClip(input)
	newclip = (clip.fx( vfx.rotate, degree))
	newclip.write_videofile(output)
	print('input file at: ', input)
	print('rotated video has been saved to : ', output)

def video_speed(input, output, speed, clipstart, clipend):
	# loading video dsa gfg intro video 
	clip = mp.VideoFileClip(input) 

	# need to add safeguards;

	# applying speed effect 
	#final = (clip.fx( vfx.speedx, speed)).subclip(0,3.1)
	final = (clip.fx( vfx.speedx, speed))
	final.write_videofile(output)
	
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

	
def slow_video(input, output, speed):
	clip = mp.VideoFileClip(input) 
	slow_clip = (clip.fx( vfx.speedx, speed))
	size = slow_clip.size
	input_list = (list(slow_clip.iter_frames()))
	length = len(input_list)
	# assert the length >= 75, otherwise pad it with anything
	# to make up to the required window width;
	window_width = 77 # leave some room: 75 + buffer;
	if(length < window_width):
		# pad it with the last frame;
		grab = input_list[-1]
		diff = window_width - length
		for i in range(0, diff):
			input_list.append(grab)
		# done padding;
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
	#print(extract)
	# now; write the extracted frame as video;
	# encode it with its bare ascii code; a wraparound;
	# src - https://forums.developer.nvidia.com/t/python-what-is-the-four-characters-fourcc-code-for-mp4-encoding-on-tx2/57701/4
	fourcc = 0x7634706d
	out = cv2.VideoWriter(output,fourcc, 30, size)
	for i in range(len(extract)):
		# writing to a image array
		out.write(extract[i])
	out.release()
	print("the slowed-video has been saved to: ", output)

if __name__ == '__main__':
	PREFIX = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\"
	input = PREFIX + "help_1.mp4"
	output = PREFIX + "output_test_3.mp4"
	input1 = "C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\ambulance_1.mp4"
	clip = mp.VideoFileClip(input)
	ls = list(clip.iter_frames())
	#print(ls[len(ls)])
	get_frame_count_CV(input)
	slow_video(input, output, speed = 0.5)
	get_frame_count_CV(output)
	

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