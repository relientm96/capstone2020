'''
Utility module to import openpose
'''

import os
import sys

# Hold parameters for OpenPose as a dictionary
params = dict()
params["net_resolution"]       = "144x144"
params["hand"]                 = True
params["hand_net_resolution"]  = "272x272"
params['keypoint_scale']       = 3
params["number_people_max"]    = 1

def initializeOpenPose():
    '''
    Initialize OpenPose for frame processing
    Returns:
        op        - pyopenpose Object
        opWrapper - OpenPose C++ Wrapper Object
    '''
    # Importing OpenPose 
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Import Models
        params["model_folder"] = "../openpose-python/models/"
        try:
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
        # Start openpose wrapper
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        print("OpenPose Wrapper Started!")
        return op, opWrapper
    except Exception as e:
        print(e)
        sys.exit(-1)