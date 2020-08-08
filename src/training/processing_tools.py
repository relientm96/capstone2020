import json
from pprint import pprint
import glob, os
import file_tools as ftool
import sign2index_dict as dict_m

import generate_XY as genxy
import json_video2txt as jv2t


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
				#print('length of body', len(upperbody_keypoints))
				print('length of left_keypoints', len(lefthand_keypoints))
				print('length of roght_keypoints', len(righthand_keypoints))
				#print('length of pose_keypoints', len(pose_keypoints))
				
			
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


def generate_XY(txt_path, path_X,  path_Y):
	'''
	args:
		1. src_path; the txt file for the converted video;
		2. path_X, where to save the X.txt;
		3. path_Y, where to save the Y.txt;
	return:
		none;
	function:
		generate the labels: Y and X for one individual video in txt format;

	'''
	#-----------------------------------------------
	# copy the individual txt to the X.txt;
	#-----------------------------------------------

	with open(path_X, 'a') as fileX, open(txt_path, 'r') as fileVideo:
		print('given txt_path\n', txt_path)
		print("Appending ... \n")
		for line in fileVideo:
			# safe guard;
			assert(line != "")
			#print("txt line, \n", line)
			fileX.write(line)
		print("finished appending\n")

	#-----------------------------------------------
	# generate its corresponding Y label for Y.txt;
	#-----------------------------------------------

	# first, update the dict in case of new entries;
	dict = dict_m.update_dict(txt_path, saved_path = "", save_name = 'saved_dict.p')

	# get the classname;
	[classname, image_checked] = ftool.checkoff_file(txt_path, "checked")
	print('classname\n', classname)
	try:
			ylabel = dict[classname]
			print("the class the image belongs to ", ylabel)
	except KeyError as e:
			print("the key doesnt exit!\n", e)	
			print("system abort\n")
			sys.exit(-1)

	# write the corresponding label to Y.txt
	with open(path_Y, 'a') as fileY:
		# add EOL to conform to txt format;
		insertline = str(ylabel) + "\n"
		fileY.write(insertline)
		print("done labelling\n")

