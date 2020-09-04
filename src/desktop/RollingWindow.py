'''
Rolling window class

created by matthew, nebulaM78 team; capstone 2020;
#ested, Fixed by Yick
'''

import numpy as np
import pprint as pp

class RollingWindow:
	def __init__(self, window_Width, numbJoints):
		# Features
		self.window_Width   = window_Width
		self.numbJoints     = numbJoints
		# Create a 2D Matrix with dimensions
		self.points         = np.zeros(shape=(window_Width,numbJoints))
		# Frame to add pointer
		self.framePointer   = 0
	
	def getPoints(self):
		# Return the 2D matrix of points
		return self.points
	
	def getWindow_Width(self):
		return self.window_Width

	def getNumbJoints(self):
		return self.numbJoints

	def printPoints(self):
		pp.pprint(self.points)

	def addPoint(self, arr):
		# Check for same number of joints in input array before we insert
		if ( len(arr) != self.numbJoints ):
			print("Error! Number of items not equal to numbJoints = ", self.numbJoints)
			return False
			
		'''
		if self.framePointer < self.window_Width:
			# If Empty, then we start by appending points at the current framePointer index
			self.points[self.framePointer] = arr
			self.framePointer +=1 

		else:
			# Not empty, so we append from last index and remove first index
			# shift register;
			# remove the oldest from the first index;
			# add the most recent entry always enters from the "last" index 
			self.points = np.delete(self.points, 0, 0)
			self.points = np.vstack([self.points, arr])

		'''
		# shift register;
		# remove the oldest from the first index;
		# add the most recent entry always enters from the "last" index 
		self.points = np.delete(self.points, 0, 0)
		self.points = np.vstack([self.points, arr])

        # Return True when everything is good
		return True

	def clearWindow(self):
		'''
		Reset all keypoints to 0's and make window append from index 0
		'''
		self.points       = np.zeros(shape=(self.window_Width,self.numbJoints))
		self.framePointer = 0 