#/usr/bin/env/python
# to "convert" image to a video of 32 frames;
# because LSTM is trained on 32 framesl
# and we treat images as videos so that we could LSTM for all;
# this is needed because the model is constructed to accept inputs of certain format;
# Created by Stuart Eiffert 13/12/2017
# modified by nebulaM78 team; capstone 2020;

# to-do; shall incorporate CNN with LSTM
# so that we have extra input (image) vector which provides
# extra information besides skeleta; to improve the inference;

import json
from pprint import pprint
import glob, os
import sys

def image2video(jsondata_path, output_path, n_frames):
	'''
	arg:
		1. jsondata_path; path where the json file is located;
		2. output_path; where to save;
		3. n_frames; number of frames;
	return:
		none;
	function:
		to "convert an image to a video of N frames 
		and reformat it to a certain format in txt;
		confer README;
	'''

	# kps is a list of lists;
	# where where each inner list is the keypoints of one frame;
	# denote N as the number of frames;
	# hence, we will have a list of N number of lists;
	kps = []
	with open(jsondata_path) as data_file: 
		
		data = json.load(data_file)
		# 'data' format:
			# {'version': xx, 'people': [{'person_id' :xx, 'pose_keypoints_2d': xx, ..., 'lefthand': xx, ...}]}
		# we are using BODY_25 model instead of COCO models; ==> 25 keypoints for the body;
		# 21 keypoints for each hand
		# get body + both hands;
		body_keypoints = data["people"][0]["pose_keypoints_2d"]
		lefthand_keypoints = data["people"][0]["hand_left_keypoints_2d"] 
		righthand_keypoints = data["people"][0]["hand_right_keypoints_2d"] 

		# safeguard;
		# empty set? for the hand keypoints?
		if (len(lefthand_keypoints) == 0):
			print("OpenPose flag for hand keypoints is OFF\n")
			print("This is not good; exit the system and check the flags;")
			sys.exit(0)

		# only need the upper body and possibly the face (?)
		# refer to how they map the keypoints here:
		# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#face-output-format
		# remember, we have (x,y accuracy) for each keypoints;
		upperbody_keypoints = body_keypoints[0:27] 
		# concatenate all the keypoints into one list: pose_keypoints;
		pose_keypoints = upperbody_keypoints + lefthand_keypoints + righthand_keypoints
		
		# ignore the confidence level;
		number_xy_coor = int((len(pose_keypoints)/3)*2)
		frame_kps = []
		j = 0
		for i in range(number_xy_coor):
			frame_kps.append(pose_keypoints[j])
			j += 1
			# recall, we have data = {x, y, confidence level}
			# ignore data[2] - confidence level;
			if ((j+1) % 3 == 0):
				j += 1

		kps.append(frame_kps)
		n_elements = len(frame_kps)
		
	# now, duplicate it n_frames times;
	# print(frame_kps)
	for i in range(1,n_frames):
		kps.append(frame_kps)
	
	
	#Now we have kps, a list of lists, that includes the x and y positions of all 18 keypoints, for all frames in the frameset
	# So a list of length frameset.length, with each element being a N-element long list.
	#Next, we simply loop through kps, writing the contents to a text file, where each sub list is a new line.
	#At this point, there is no overlap, and datasets are all of varying length
	with open(output_path, "w") as text_file:
		for i in range(len(kps)):
			for j in range(n_elements):
				text_file.write('{}'.format(kps[i][j]))
				if j < (n_elements-1):
					text_file.write(',')
			text_file.write('\n')

# test driver;
if __name__ == '__main__':
	# paths;
	data_path = "C:\\Users\\yongw4\\Desktop\\yick\\utilities\\test_keypoints.json"
	output_path = "C:\\Users\\yongw4\\Desktop\\yick\\utilities\\test.txt"
	n_frames = 32
	image2video(data_path, output_path, n_frames)