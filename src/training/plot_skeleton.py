# acknowledgement - https://abouelnaga.io/projects/research-ops-running-open-pose/
# author - Yehya Abouelnaga
# modified/commented by - yick
# purpose 
# - to visualize the json data keypoints for sanity check for preprocessing and data augmentation;


import scipy.io
from tqdm import tqdm   
# tqdm, provide progress bar;
# They provide you with an estimate of the progress that has been made and an approximation of the time it might take more
# convenient when processing huge datasets;
# info - https://tqdm.github.io/

from typing import List, Dict, Any
import json
import matplotlib.pyplot as plt
from numpy import random

# label the json keypoints for convenience;
# global variable
OpenPoseMap = [
	"Nose",
	"Neck",
	"RShoulder",
	"RElbow",
	"RWrist",
	"LShoulder",
	"LElbow",
	"LWrist",
	"MidHip",
	"RHip",
	"RKnee",
	"RAnkle",
	"LHip",
	"LKnee",
	"LAnkle",
	"REye",
	"LEye",
	"REar",
	"LEar",
	"LBigToe",
	"LSmallToe",
	"LHeel",
	"RBigToe",
	"RSmallToe",
	"RHeel",
	"Background",
]

# global variable
# optional;
SkeletonFixedPoints =[
	0, # Nose
	1, # Neck
	2,3,4, # Right Shoulder / Elbow / Wrist
	5,6,7, # LEft shoulder / Elbow / Wrist
	8, # mid hip
	9, 10, 11, # right hip / knee / ankle
	12, 13, 14 # left hip / knee / ankle
]

# global variable
SkeletonBones = [
	[0, 1], 		# Nose - Neck
	[1, 2], 		# Neck -> Right Shoulder
	[2, 3], 		# Right Shoulder -> Right Elbow
	[3, 4], 		# Right Elbow -> Right Wrist
	[1, 5], 		# Neck -> Left Shoulder
	[5, 6], 		# Left Shoulder -> Left Elbow
	[6, 7], 		# Left Elbow -> Left Wrist
	[1, 8], 		# Neck -> Mid Hip
	[8, 9], 		# Mid hip -> Right Hip
	[9, 10], 		# Right Hip -> Right Knee
	[10, 11], 		# Right Knee -> Right Ankle
	[8, 12], 		# Mid hip -> Left hip
	[12, 13], 		# left hip -> left knee
	[13, 14], 		# Left knee -> left ankle
	# uncomment the following for aesthetic reasons;
	[0, 15], 		# nose -> right eye
	[0, 16], 		# nose -> left eye
	[15, 17], 		# right eye -> right ear
	[16, 18], 		# left eye -> left ear
]

def batch(iterable, n=1):
	'''
	args - iterable;
	returns - a generator;
	purpose - to reduce overhead as we might process huge no. of json files from videos;
	info - # info - https://wiki.python.org/moin/Generators
	'''
	length = len(iterable)
	for index in range(0, length, n):
		yield iterable[index : min(index + n, length)]

# '->' here is a function annotation;
# info - https://www.python.org/dev/peps/pep-3107/
def read_openpose_json(filename: str, add_noise) -> List[Dict[str, Any]]:
	'''
	args -
		- json filename;
		- add_noise;
			- no noise if 0;
	returns - 
		list containing - 
			- x,y 2d keypoint coordinates along with its 'c' confidence;
			- the associated body part;
			- the associated index (for convenience);
	'''
	with open(filename, "rb") as file:
		keypoints_list = []
		keypoints = json.load(file)

		# safeguarding;
		assert (
			len(keypoints["people"]) == 1
		), "In all pictures, we should have only one person!"

		# we only care about 2d; ignore the rest in the json;
		points_2d = keypoints["people"][0]["pose_keypoints_2d"]
		assert (
			len(points_2d) == 25 * 3
		), "We have 25 points with (x, y, c); where c is confidence."

		# safe? process them;
		for point_index, (x, y, confidence) in enumerate(batch(points_2d, 3)):
			assert x is not None, "x should be defined"
			assert y is not None, "y should be defined"
			assert confidence is not None, "confidence should be defined"
			if(add_noise):
				# loc = mean;
				# scale = standard deviation
				noise = random.normal(loc = 0, scale = 1)
			else:
				noise = 0
			keypoints_list.append(
				{
					"x": x + noise,
					"y": y + noise,
					"c": confidence,
					"point_label": OpenPoseMap[point_index],
					"point_index": point_index,
				}
			)
		return keypoints_list

def draw_skeleton(keypoints_list):
	'''
		args - the output from read_openpose_json()
		returns - none
		task - reconstruct the (upside-down) skeleton from json keypoints;
	'''
	print('hello, my skeleton friend, welcome to the world!')
	
	# draw the bones;
	for i in range(len(SkeletonBones)):
		p1 = SkeletonBones[i][0]
		p2 = SkeletonBones[i][1]
		x_values = [keypoints_list[p1]['x'], keypoints_list[p2]['x']]
		y_values = [keypoints_list[p1]['y'], keypoints_list[p2]['y']]
		plt.plot(x_values, y_values)
	plt.axis('equal')
	plt.show()
	return None

def compare_skeletons(clean_input, noisy_input):
	'''
		args - two sets of output from read_openpose_json()
			- clean;
			- added with gaussian noise;
		returns - none
		task - compare both skeletons visually as only one figure object
			is supported when run in the interactive command line which 
			renders  draw_skeleton() useless...
	'''
	 # draw the clean bones;
	fig = plt.figure()
	plt.subplot(1,2,1)
	for i in range(len(SkeletonBones)):
		p1 = SkeletonBones[i][0]
		p2 = SkeletonBones[i][1]
		x_values = [clean_input[p1]['x'], clean_input[p2]['x']]
		y_values = [clean_input[p1]['y'], clean_input[p2]['y']]
		plt.plot(x_values, y_values)
	
	# draw the dirty bones;
	plt.subplot(1,2,2)
	for i in range(len(SkeletonBones)):
		p1 = SkeletonBones[i][0]
		p2 = SkeletonBones[i][1]
		x_values = [noisy_input[p1]['x'], noisy_input[p2]['x']]
		y_values = [noisy_input[p1]['y'], noisy_input[p2]['y']]
		plt.plot(x_values, y_values)
	plt.axis('equal')
	plt.xlabel('x')
	plt.ylabel('y')
	fig.suptitle('LHS = clean; RHS = noisy')
	plt.show()
	return None

# test driver;
if __name__ == '__main__':
	# source: capstone2020/src/data/json-data/matthewData/handwave/json/000000000013_keypoints.json
	# change it to your location when testing this script;
	#filestr = 'C:\\Users\\yongw4\\Desktop\\test-ffmpeg\\output_test_000000000000_keypoints.json'
   
	filestr2 = 'C:\\Users\\yongw4\\Desktop\\DUMMY_JSON\\output_test_000000000071_keypoints.json'
	#filestr2 = 'C:\\Users\\yongw4\\Desktop\\DUMMY_JSON\\translation_centre_000000000006_keypoints.json'

	filestr = filestr2
	clean_output = read_openpose_json(filestr, 0)   
	augment_output = read_openpose_json(filestr2, 0)

	# check the output from the terminal;
	print(clean_output)
	print(augment_output)
	
	# oops, the skeleton is upside down ...
	draw_skeleton(clean_output)
	draw_skeleton(augment_output)

	# comparing both skeletons: clean vs noisy;
	compare_skeletons(clean_output, augment_output)
