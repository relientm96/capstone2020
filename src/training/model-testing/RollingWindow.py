
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
		self.window_Width = window_Width
		self.numbJoints   = numbJoints
		# Create a 2D Matrix with dimensions
		self.points       = np.zeros(shape=(window_Width,numbJoints))
		# Frame to add pointer
		self.isReset 	  = True
	
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

		if self.isReset == True:
			# Empty, so we duplicate this point across all keypoints in window
			self.points = np.array([arr for x in range(self.window_Width)])
			self.isReset = False
		else:			
			# shift register;
			# remove the oldest from the first index;
			# add the most recent entry always enters from the "last" index 
			self.points = np.delete(self.points, 0, 0)
			self.points = np.vstack([self.points, arr])

        # Return True when everything is good
		return True

	def resetWindow(self):
		'''
		Reset keypoints to duplicate all across all indexes
		'''
		self.isReset = True