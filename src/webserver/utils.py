'''
Module for Numpy Utility Functions
'''

import numpy as np

def removeConfidenceAndShapeAsNumpy(datum):
    '''
    Takes in datum object and get pose, and both hand keypoints 
    and returns a numpy array for prediction
    '''
    # Test and see if we have anyone in image
    try:
        test = len(datum.poseKeypoints[0])
    except Exception as e:
        # Test will return an error if no one is seen as index[0] does not exists
        # Notify user that no one is seen
        return np.zeros(1)

    # We want to keep only first two columns for all keypoints (ignore confidence levels)
    posePoints = datum.poseKeypoints[0][1:8,0:2]  # Slice to only take keypoints 1-7 removing confidence
    lefthand   = datum.handKeypoints[0][0][:,0:2] # Get all hand points removing confidence
    righthand  = datum.handKeypoints[1][0][:,0:2] # Get all hand points removing confidence

    # Concatenate all numpy matrices and flatten for rolling window storage (as a row)
    keypoints = np.vstack([posePoints,lefthand,righthand]).flatten()
    
    # Return this as the keypoint to be added to rolling window
    return keypoints