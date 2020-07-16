# For OpenPose Module
import sys
import cv2
import os
from sys import platform
import argparse
import time

# Initializing OpenPose Python Wrapper Globally
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
	# Windows Import
	if platform == "win32":
		sys.path.append(dir_path + '/../openpose-python/Release')
		os.environ['PATH']  = os.environ['PATH']  + ';' +  dir_path + "/../openpose-python" + ';' + dir_path + "/../openpose-python/bin" 
		import pyopenpose as op
except ImportError as e:
	print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
	raise e
except Exception as e:
	print(e)
	sys.exit(-1)

#################### Custom Params for OpenPose ######################
params = dict()
params["model_folder"] = "../openpose-python/models/"

# ONLY CHANGE BELOW (configure if needed)
params["net_resolution"] = "96x96"
#params['write_json'] = "C:\\CAPSTONE\\capstone2020\\src\\training"
params["hand"] = True
#params["hand_net_resolution"] = "328x328"
######################################################################

# Preprocessing data for training our models
from pathlib import Path
import shutil

# ========================= Object Classes ========================= #

class OpenPoseProcessor:
	def __init__(self):
		'''
		Initializes flags and starts OpenPose Wrapper
		'''
		# Flags
		parser = argparse.ArgumentParser()
		self.args = parser.parse_known_args()
		# Add other flags in path (if available)
		for i in range(0, len(self.args[1])):
			curr_item = self.args[1][i]
			if i != len(self.args[1])-1: next_item = self.args[1][i+1]
			else: next_item = "1"
			if "--" in curr_item and "--" in next_item:
				key = curr_item.replace('-','')
				if key not in params:  params[key] = "1"
			elif "--" in curr_item and "--" not in next_item:
				key = curr_item.replace('-','')
				if key not in params: params[key] = next_item

	def run_openpose(self, imagepath, outputDirPath):
		'''
		Runs OpenPose on an image given imagepath
		@params: imagepath = image path of file to be processed
		@params: outputDirPath = directory of output file's keypoints
		@returns datum: keypoints as a datum object   
		Access keypoints outside this function as such:
			datum.poseKeypoints[0] --> Pose Keypoints
			datum.handKeypoints[0] --> Left Hand Keypoints
			datum.handKeypoints[1] --> Right Hand Keypoints
		'''
		# Adjust output flag
		params['write_json'] = outputDirPath
		# Starting OpenPose for each new flag
		self.opWrapper = op.WrapperPython()
		self.opWrapper.configure(params)
		self.opWrapper.start()
		# Process Image
		datum = op.Datum()

		# change the output json filename for convenience;
		# assume the image path has only one separator, i.e. "xxx.jpg"
		filename = imagepath.split('.')[0]

		if platform == "win32":
			datum.name = filename.split('\\')[-1]
		else:
			datum.name = filename.split('/')[-1]

		imageToProcess = cv2.imread(imagepath)
		datum.cvInputData = imageToProcess
		self.opWrapper.emplaceAndPop([datum])

		# Processes image and sends back keypoints via the datum object
		return datum

# ======================================================================================== #

class DataPreprocessor:
	def __init__(self):
		self.data_path = Path('../data')
		self.keypoints_path = self.data_path / 'keypoints'
		self.raw_data_path = self.data_path / 'raw-data'
		self.op_proc = OpenPoseProcessor()

	def process_data(self):
		self._make_destination_dir()
		self._images_to_keypoints()

	def _make_destination_dir(self):
		'''
		make destination directory to store keypoints (output from openpose)
		'''
		if not self.keypoints_path.exists():
			self.keypoints_path.mkdir()

		self.img_paths = [p for p in self.raw_data_path.rglob('*.png')]
		img_rel_paths = [p.relative_to(self.raw_data_path).parent for p in self.img_paths]
		self.dst_paths = [self.keypoints_path / rp for rp in img_rel_paths]

		for dst in self.dst_paths:
			if not dst.exists():
				dst.mkdir(parents=True)

	def _images_to_keypoints(self):
		'''
		run openpose on the images one by one
		'''
		for src, dst_dir in zip(self.img_paths, self.dst_paths):
			self._run_openpose(src, dst_dir)

	def _run_openpose(self, src, dst_dir):
		'''
		run openpose on one source (just one image for now)
		params: src is the relative path to the image file (including the file)
				dst_dir is the relative path to the destination file (excluding the output file)
		TODO: it maybe more convenient to use absolute path but we will see.
		TODO: maybe we can use mongodb
		'''
		# TODO: at the moment the name of the json file is not specified (although the location is specified).
		# We might have to figure out a way to specify the name.
		print(str(src))
		self.op_proc.run_openpose(str(src), str(dst_dir))

if __name__ == '__main__':
	pp = DataPreprocessor()
	pp.process_data()

	#### EXAMPLE PROGRAM ####
	# Create OpenPoseProcessor Object
	opProc =  OpenPoseProcessor()
	# Define a list of image paths for test purposes
	#listOfImages = ["test1.jpg","test2.jpg"]
	listOfImages = ["test1.jpg", 'test2.jpg']

	for imagepath in listOfImages:
		# stringOfPath = imagepath.split('.')[0]
		# Run and print keypoints for each image in listOfImages
		stringOfPath = "C:\\CAPSTONE\\capstone2020\\src\\training\\test1\\"
		output = opProc.run_openpose(imagepath, stringOfPath)
		print(output)
		print("Pose keypoints\n", output.poseKeypoints[0])
		print("Left hand keypoints\n", output.handKeypoints[0])
		print("Right hand keypoints\n", output.handKeypoints[1])