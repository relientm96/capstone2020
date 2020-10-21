'''
Main Script used in generating plot/result data from videos

'''
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
print("Tensorflow Version:", tf.__version__)

# Custom openpose import
import openpose

def main():
    '''
    Hello world Yick, use this to do testing

    Set your openpose configs in openpose.py
    '''
    cap = cv2.VideoCapture('test-vid-dir/pain_1.mp4')
    # Openpose Reference Objects
    op, opWrapper = openpose.initializeOpenPose()

    if cap.isOpened() == False:
        print("Error Opening Video file!")

    while(cap.isOpened()):
        # Reading a single frame from video opened by cap
        ret, frame = cap.read()
        if ret == True:
            #cv2.imshow('Frame', frame)
            # Now read vid frame into openpose
            datum = op.Datum()
            imageToProcess = cv2.imread(frame)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            # Get pose outputs
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
            cv2.imshow("For Yick to see", datum.cvOutputData)
            if cv2.waitKey(-1) & 0xFF == ord('q'):
                break
    # Clean up after done
    cap.release()
    cap.destroyAllWindows()

if __name__ == "__main__":
    main()