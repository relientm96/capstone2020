#/usr/bin/env/python
# created by matthew, nebulaM78 team; capstone 2020;
# unit testing for the relevant module in gestureRecognition.py and serverOpenPose.py
# 1. removeConfidenceAndAppend()

import sys
import cv2
import os
from sys import platform
import argparse
import errno
import shutil
import json

def removeConfidenceAndAppend(data):
	'''
	args  - json datum
	function -  to remove confidence levels and append to keypoint list
	output - reformatted list containing all the x.y coordinate of certain keypoints;
	'''
	# only need the upper body and possibly the face (?)
	# refer to how they map the keypoints here:
	# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#face-output-format
	# remember, we have (x,y accuracy) for each keypoints;
	
	# safe guard;
	if(len(data['people']) == 0):
		print("at current frame, no people are detected, so skipped\n")
		# a total of 147 keypoints;
		# initialized to zero;
		pose_keypoints = [0]*147
	else:
		body_keypoints = data["people"][0]["pose_keypoints_2d"]
		# based on BODY25, only get points indexed from 1-7;
		upperbody_keypoints = body_keypoints[3:24]
		# get all the hand keypoints;
		lefthand_keypoints = data["people"][0]["hand_left_keypoints_2d"] 
		righthand_keypoints = data["people"][0]["hand_right_keypoints_2d"] 
		pose_keypoints = upperbody_keypoints + lefthand_keypoints + righthand_keypoints
	# pose keypoints are done processed;
	# ignore the confidence level;
	number_xy_coor = int((len(pose_keypoints)/3)*2)
	outputlist = []
	j = 0
	for i in range(number_xy_coor):
		outputlist.append(pose_keypoints[j])
		j += 1
		# recall, we have data = {x, y, confidence level}
		# ignore data[2] - confidence level;
		if ((j+1) % 3 == 0):
			j += 1
	return outputlist


# test driver;
if __name__ == '__main__':
	#------------------------------------
	# removeConfidenceAndAppend()
	#------------------------------------
	filepath = "C:\\Users\\yongw4\\Desktop\\JSON\\testing_000000000353_keypoints.json"
	with open(filepath) as jsonfile:
		keypoints = json.load(jsonfile)
		#kp_pose = []
		#kp_lefthand = []
		#kp_righthand = []
		#kp = []
		# #  test 01
		# # check the size;
		#kp_pose = removeConfidenceAndAppend(keypoints['people'][0]['pose_keypoints_2d'], 24, kp_pose)
		#kp_lefthand = removeConfidenceAndAppend(keypoints['people'][0]['hand_left_keypoints_2d'],  63, kp_lefthand)
		#kp_righthand = removeConfidenceAndAppend(keypoints['people'][0]['hand_right_keypoints_2d'], 63,kp_righthand)              
		#print('len pose', len(kp_pose))
		#print('len lefthand', len(kp_lefthand))
		#print('len righthand', len(kp_righthand))
		
		# # test 02
		# # check against the real json file;
		kp = removeConfidenceAndAppend(keypoints)
		print('len kp', len(kp))
		print('output', kp)