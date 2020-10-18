'''
Module for Numpy Utility Functions
'''
import numpy as np


def offset_translation(array, reference):
    '''
    purpose: for translational invariance;
    Args:
            array; must be in the  shape of (1, number_of_joints, 3);
            refrence: the refrence to be subtracted from; presumable the shoulder;
    Returns:
            Shifted array (all points subtracted from reference)
    '''
    broadcast = np.array([reference, 0, 0], dtype=np.float32)
    shifted = array - broadcast
    return shifted


def removeConfidenceAndShapeAsNumpy(datum):
    '''
    Takes in datum object and get pose, and both hand keypoints 
    and returns a numpy array for prediction
    '''
    # Test and see if we have anyone in image
    try:
        test = len(datum.poseKeypoints[0])
        # translational invariant from shoulder center
        shoulder_center = datum.poseKeypoints[0][3][0]
        body_keypoints = offset_translation(
            datum.poseKeypoints, shoulder_center)
        lefthand_keypoints = offset_translation(
            datum.handKeypoints[0], shoulder_center)
        righthand_keypoints = offset_translation(
            datum.handKeypoints[1], shoulder_center)

        # We want to keep only first two columns for all keypoints (ignore confidence levels)
        # Slice to only take keypoints 1-7 removing confidence
        posePoints = body_keypoints[0][1:8, 0:2]
        # Get all hand points removing confidence
        lefthand = lefthand_keypoints[0][:, 0:2]
        # Get all hand points removing confidence
        righthand = righthand_keypoints[0][:, 0:2]

        # Concatenate all numpy matrices and flatten for rolling window storage (as a row)
        keypoints = np.vstack([posePoints, lefthand, righthand]).flatten()

        # Return this as the keypoint to be added to rolling window
        return keypoints

    except Exception as e:
        # Test will return an error if no one is seen as index[0] does not exists
        # Notify user that no one is seen
        print(str(e))
        return np.zeros(1)
