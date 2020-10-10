import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pprint as pp


def PARAMS(jsonpath = "JSON"):
	# OpenPose Configurations
	params = dict()
	params["net_resolution"]       = "160x160"
	params["hand_net_resolution"]  = "256x256"
	params["hand"]                 = True
	params['keypoint_scale']       = 3
	params["disable_multi_thread"] = False
	#params["number_people_max"]    = 1
	params["write_json"] = jsonpath
	return params

# Reference Objects
op = None
opWrapper = None

def importOpenPose():
	params = PARAMS()
		
	'''
	Import OpenPose Library and wrapper
	'''

	try:
		dir_path = os.path.dirname(os.path.realpath(__file__))
		params["model_folder"]         = "../openpose-python/models/"
		try:
			# Windows Import
			if platform == "win32":
				# Change these variables to point to the correct folder (Release/x64 etc.)
				sys.path.append(dir_path + '/../openpose-python/Release')
				os.environ['PATH']  = os.environ['PATH']  + ';' +  dir_path + "/../openpose-python" + ';' + dir_path + "/../openpose-python/bin" 
				import pyopenpose as op
		except ImportError as e:
			print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
			raise e
		except Exception as e:
			print(e)
			sys.exit(-1)

		# Flags
		parser = argparse.ArgumentParser()
		parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
		parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
		args = parser.parse_known_args()

		# Add others in path?
		for i in range(0, len(args[1])):
			curr_item = args[1][i]
			if i != len(args[1])-1: next_item = args[1][i+1]
			else: next_item = "1"
			if "--" in curr_item and "--" in next_item:
				key = curr_item.replace('-','')
				if key not in params:  params[key] = "1"
			elif "--" in curr_item and "--" not in next_item:
				key = curr_item.replace('-','')
				if key not in params: params[key] = next_item

		# Start openpose wrapper
		opWrapper = op.WrapperPython()
		opWrapper.configure(params)
		opWrapper.start()
		print("OpenPose Wrapper Started!")
		return op, opWrapper

	except Exception as e:
		print(e)
		sys.exit(-1)

def run_directory():
	# Import openpose
	op, opWrapper = importOpenPose()

	# Read images from a directory
	image_directory_path = 'test-images/augment-3'
	imagePaths = op.get_images_on_directory(image_directory_path)
	# Process and display images
	for imagePath in imagePaths:
		print("imagepath: ", imagePath)
 
		print("Processing Image {} from directory {}".format(imagePath, image_directory_path))
		datum = op.Datum()
	
		imageToProcess = cv2.imread(imagePath)
		datum.cvInputData = imageToProcess
        # tweak on how openpose name the json output;
		#datum.name = "image"
		opWrapper.emplaceAndPop([datum])
		print("Body keypoints: \n" + str(datum.poseKeypoints))
		print("Left Hand keypoints: \n" + str(datum.handKeypoints[0]))
		print("Right Hand keypoints: \n" + str(datum.handKeypoints[1]))
		# Display Image
		cv2.imshow("Sample Image Run using OpenPose", datum.cvOutputData)
		cv2.waitKey(0)

'''
tmp = (imagePath.split("\\")[-1])
		tmp = tmp.split(".")[0]
		jsonpath = tmp+".json"
		print('jsonpath:', jsonpath)
		params["write_json"] =  jsonpath
'''

if __name__ == "__main__":
	run_directory()

