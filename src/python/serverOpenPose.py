# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time

# Own Library
import gestureRecognition as gr

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Initialize Gesture Recognition Program
    gr.initOpenPoseLoad()

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Get input from webcam
    cap = cv2.VideoCapture(0)
    while True:
        try:
            # Read image from webcam
            ret, frame = cap.read()
            # Send image to OpenPose for processing
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # Pass in datum object to send keypoints to gesture recognition module
            word = "Word: " + gr.translate(datum)

            # Adding all of these into image
            image = datum.cvOutputData
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (50, 50) 
            fontScale = 1
            color = (255, 255, 0) # Color chosen from BGR values (0-255) 
            thickness = 2
            image = cv2.putText(image, word, org, font,  
                            fontScale, color, thickness, cv2.LINE_AA)
                    
            # Display Rendered Poses + Word On OpenCV GUI
            cv2.imshow("CAPSTONE 2020, Sign Language Translation System", image)
            # Wait 1ms after processing to display image to OpenCV2's GUI Window
            cv2.waitKey(1)

        except Exception as e:
            print(e)
            sys.exit(-1)

except Exception as e:
    print(e)
    sys.exit(-1)
