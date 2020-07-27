#/usr/bin/env/python
# to reformat json files and save them as a txt for training or inference;
# this is needed because the model is constructed to accept inputs of certain format;
# Created by Stuart Eiffert 13/12/2017
# modified by nebulaM78 team; capstone 2020;

import json
from pprint import pprint
import glob, os

def json_video2txt(jsondata_path, output_path):
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
			
			# debugging ...
			#print("json_video2txt-debug\n")
			#print('data\n', data)
			#assert(len(data["people"]) != 0)
			
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
				
				# debugging ...
				#print("json_video2txt-1\n")
			
				upperbody_keypoints = body_keypoints[3:24] 
			
				# debugging ..
				#print("json_video2txt-2\n")

				# concatenate all the keypoints into one list: pose_keypoints;
				pose_keypoints = upperbody_keypoints + lefthand_keypoints + righthand_keypoints
			
				#print(len(pose_keypoints))
			
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
	
	data_path = "C:\\Users\\yongw4\\Videos\\basic\\testing" + "\\dummy_02"

	output_path = "C:\\Users\\yongw4\\Videos\\basic\\testing" + "\\dummy_02.txt"
	json_video2txt(data_path, output_path)