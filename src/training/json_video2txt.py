#/usr/bin/env/python
# to reformat json files and save them as a txt for training or inference;
# this is needed because the model is constructed to accept inputs of certain format;
# Created by Stuart Eiffert 13/12/2017
# modified by nebulaM78 team; capstone 2020;

import json
from pprint import pprint
import glob, os

# if the confidence value for the keypoint is very low;
# no point using it;
def nullify_keypoints(input_list):
	ls = input_list
	length = len(input_list)
	fixed = 2
	for index, elem in enumerate(input_list):
		if ((abs(index - fixed)%3) == 0):
			confidence = input_list[index]
			if(confidence <= 0.1):
				ls[index-1] = -1
				ls[index-2] = -1
	return ls

def offset_translation(input_list, reference):
	ls = [(x-reference) if not(i%3) else x for i, x in enumerate(input_list)]
	return ls

def json_video2txt(jsondata_path, output_path, func):
	'''
	arg:
		1. jsonpath; data path where the set of json files located;
	   (one set has N-frame json files corresponding to one class)
	   2. output_path; where to save;
	return:
		none;
	function:
		1. convert the json files to certain format and save it as txt;
	'''

	# kps is a list of lists;
	# where where each inner list is the keypoints of one frame;
	# denote N as the number of frames;
	# hence, we will have a list of N number of lists;
	kps = []
	jsondata_path = jsondata_path + "\\*.json"
	for file in sorted(glob.glob(os.path.join(jsondata_path))):
		with open(file) as data_file: 
		
			data = json.load(data_file)
			# 'data' format:
				# {'version': xx, 'people': [{'person_id' :xx, 'pose_keypoints_2d': xx, ..., 'lefthand': xx, ...}]}
			# we are using BODY_25 model instead of COCO models; ==> 25 keypoints for the body;
			# 21 keypoints for each hand
			# get body + both hands;
			
			# in case of no people detected, 
			# hence, we will have empty sets; 
			# skip
			if(len(data['people']) == 0):
				print("at current frame, no people are detected, so skipped\n")
				# a total of 147 keypoints;
				# initialized to zero;
				pose_keypoints = [0]*147
			# normal circumstance;
			else:
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
				
				# centering with respect to the shoulder;
				# translation invariant;
				shoulder_center = body_keypoints[3]
				body_keypoints = offset_translation(body_keypoints, shoulder_center)
				lefthand_keypoints = offset_translation(lefthand_keypoints, shoulder_center)
				righthand_keypoints = offset_translation(righthand_keypoints, shoulder_center)

				# nullify the keypoint to a fixed constant if 
				# its associated confidence level is very low (<= 0.1);
				body_keypoints  = nullify_keypoints(body_keypoints)
				lefthand_keypoints  = nullify_keypoints(lefthand_keypoints)
				righthand_keypoints  = nullify_keypoints(righthand_keypoints)

				# do some extra stuff with the function passed;
				body_keypoints  = func(body_keypoints)
				lefthand_keypoints  = func(lefthand_keypoints)
				righthand_keypoints  = func(righthand_keypoints)

				upperbody_keypoints = body_keypoints[3:24] 
				# concatenate all the keypoints into one list: pose_keypoints;
				pose_keypoints = upperbody_keypoints + lefthand_keypoints + righthand_keypoints
				
			# pose keypoints are done processed;
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
			#print('number of xy',number_xy_coor)
			#print('leng of frame_kps', len(frame_kps))
			kps.append(frame_kps)
			n_elements = len(frame_kps)
			
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
	
	PATH = "C:\\Users\\yongw4\\Desktop\\JSON\\"
	data_path = PATH 

	output_path = PATH + "\\X_test.txt"
	#json_video2txt(data_path, output_path)

	input_list = [10,3,3, 10,2,2,10,4,1]
	#print(offset_translation(input_list, 1))

	input_list = [10,3,1, 10,2,0.01,10,4,0.002]
	#print(nullify_keypoints(input_list))

	