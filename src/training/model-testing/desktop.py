# created by matthew;
# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import re
import numpy as np

# Own Library
import gestureRecognition as gr

# Hold parameters for OpenPose as a dictionary
params = dict()

params["disable_blending"]     = False

params["net_resolution"]       = "128x128"
params["hand_net_resolution"]  = "256x256"

#params["net_resolution"]      = "336x336"
#params["hand_net_resolution"] = "328x328"
#params["disable_blending"]    = True

params["hand"]                 = True
params['keypoint_scale']       = 3
#params["disable_multi_thread"] = False
params["number_people_max"]    = 1
############################################

cap = cv2.VideoCapture(0)

# Importing OpenPose 
try:
    '''
    Import OpenPose Library and wrapper
    '''
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # different dir path;
    dir_path = "C:\\CAPSTONE\\capstone2020\\src\\training"
    print("dir_path: ", dir_path)
    # Import Models
    params["model_folder"] = "/../openpose-python/models/"
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../openpose-python/Release')
            os.environ['PATH']  = os.environ['PATH']  + ';' +  dir_path + "/../openpose-python" + ';' + dir_path + "../../openpose-python/bin" 
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

    # Import OpenPose library to Gesture System
    gr.initOpenPoseLoad()
    print("Gesture Recognition System Started!")

    # Start openpose wrapper
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    print("OpenPose Wrapper Started!")

    # Load Model from keras
    gr.loadModel()
    print('Successfully Loaded Model')

except Exception as e:
    print(e)
    sys.exit(-1)

def main():
   
    while (True):
        try:
            # Read capture from opencv2
            ret, frame = cap.read()

            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # Pass in datum object to send keypoints to gesture recognition module
            word = gr.translate(datum)

            # Adding all of these into image
            image = datum.cvOutputData
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (50, 50) 
            fontScale = 1
            color = (255, 255, 0) # Color chosen from BGR values (0-255) 
            thickness = 2
            image = cv2.putText(image, word, org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)

            image = cv2.resize(image, (1024,768))
            # Show result
            cv2.imshow('frame',image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except Exception as e:
            print(e)
            
    # Break and release if detected
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()