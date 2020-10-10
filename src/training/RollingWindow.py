#/usr/bin/env/python
# created by matthew, nebulaM78 team; capstone 2020;
# unit testing for the relevant class in gestureRecognition.py and serverOpenPose.py
# 1. RollingWindow()

import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pprint as pp
import shutil
import json
import removeConfidenceAndAppend as RCA # make sure it's in the same directory;

class RollingWindow:
	def __init__(self, window_Width, numbJoints):
		self.window_Width   = window_Width
		self.numbJoints     = numbJoints
		# Create a 2D Matrix with dimensions
		# window_Width X number of joints
		self.points         = np.zeros(shape=(window_Width,numbJoints))
	
	def getPoints(self):
		# Return the 2D matrix of points
		return self.points
	
	def getWindow_Width(self):
		return self.window_Width

	def getNumbJoints(self):
		return self.numbJoints

	def addPoint(self, arr):
		# Check for same number of joints in input array before we insert
		if ( len(arr) != self.numbJoints ):
			print("Error! Number of items not equal to numbJoints = ", self.numbJoints)
			return False
		
		# shit register;
		# remove the oldest from the first index;
		# add the most recent entry always enters from the "last" index 
		self.points = np.delete(self.points, 0, 0)
		self.points = np.vstack([self.points, arr])
		
		return True

	def printPoints(self):
		pp.pprint(self.points)

	def clearWindow(self):
		self.points = np.zeros(shape=(self.window_Width,self.numbJoints))

# test driver;
if __name__ == '__main__':
	#------------------------------------
	# rolling window
	#------------------------------------
	window_Width = 2
	numbJoints = 5
	print("Creating Rolling Window")
	rolling_window = RollingWindow(window_Width,numbJoints)
	print("Finished Created Rolling Window, Window Width = {} & NumbJoints = {}".format(window_Width, numbJoints))
	print('initialized rolling window ', rolling_window.getPoints())
	
	# # test 01: add the frames sequentially;
	for i in range(1,7):
		kp = [i]*numbJoints
		rolling_window.addPoint(kp)
		print('added the new frame, \n ', rolling_window.getPoints())
	
	# clear the window;
	print('final window, \n ', rolling_window.getPoints())
	print('clearing the window:')
	rolling_window.clearWindow()
	print('has it been cleared? ', rolling_window.getPoints())


	