#/usr/bin/env/python
# to start window to record video;
# adapted and modified by nebulaM78 team; capstone 2020;
import cv2
import time
import sys
import numpy as np
# needed to save non-string python object; eg dictionary;
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle


# src = https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# src - https://raspberrypi.stackexchange.com/questions/66976/capture-video-for-a-certain-time-then-quit-and-save-to-a-folder-using-opencv-3
# src - https://www.geeksforgeeks.org/python-opencv-write-text-on-video/
# src - https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get

def record_video(filename, signtime = 3, preptime = 3):
	'''
	args:
		- filename; where to save the recorded video?
		- signtime; how long do you want to capture the actual sign;
		- preptime; how long do you need to prepare before signing;
	return:
		- None;
	function:
		- start up a self recording window and save it;
	'''
	# capture frames from a camera with device index=0
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	cap = cv2.VideoCapture(0)
	fps = 20    # h0ow many frames you want to write in a second;
	print('writing fps: ', fps)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

	# Define the duration (in seconds) of the video capture here
	#preptime = 3
	buffertime = 0.3
	#signtime = 3
	capture_duration = preptime + signtime + buffertime
	start_time = time.time()
		
	while( int(time.time() - start_time) < capture_duration ):
	
		# reads frame from a camera 
		_, frame = cap.read() 
	
		# describe the type of font 
		# to be used. 
		font = cv2.FONT_HERSHEY_SIMPLEX 
  
		# give a head up to the user before the actual signing;
		# no need to save the frame within this interval;
		tick = int(time.time() - start_time)
		if(tick <= preptime):
			TEXT = "Start signing in " + str(tick)
			cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4)
			cv2.imshow('Camera',frame) 

		# the user should be prepared now;
		else:
			TEXT = "OOO, Handsome! " + "tick: " + str(tick-preptime)
			cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4) 
 
			# Display the frame
			cv2.imshow('Camera',frame) 
			# write the frames here to the file;
			out.write(frame)
			# user could force shut down by pressing "Q"
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		end_recordtime = time.time()

	# sanity check on the total time taken;
	print('sanity check:\n total time taken: ', end_recordtime - start_time)
	print('whole capture duration: ', capture_duration)
	
	# release the camera from video capture
	cap.release() 
	
	# De-allocate any associated memory usage 
	cv2.destroyAllWindows() 

	print('the video has been saved to:', filename)

def annotate_video(filename, info_list):
	'''
	args:
		- filename; where to load the saved video?
	return:
		- None;
	function:
		- edit the saved video but will not save the edited version;
		- this serves as auxiliary tool for sanity checking;
	'''
	cap = cv2.VideoCapture(filename)
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	input_fps = int(cap.get(5))
	total_frames = int(cap.get(7))

	print('input fps: ', input_fps)
	print('number of frames in the input video: ', total_frames)
	stepsize = 15
	future_index = 0

	count_frame = 1
	track = 0
	while(cap.isOpened()):
		# prevent out-of-range error;
		if not (future_index > (len(info_list)-1)):
			TEXT = str(info_list[future_index])
			font = cv2.FONT_HERSHEY_SIMPLEX 

		# update the sign every 0.5 second (15 frames);
		if((count_frame % stepsize) == 0):
			future_index += 1
			track += 1
		# process the frame;
		ret, frame = cap.read()
		#print(np.shape(frame))
		cv2.putText(frame, TEXT, (50, 400),  font, 1,  (0, 255, 255), 2, cv2.LINE_4) 
 
		cv2.imshow('frame', frame)
		# set it to "30" to slow down the playback;
		# note: the higher this value is, the slower the playback is;
		if cv2.waitKey(input_fps*2) & 0xFF == ord('q'):
			break
		count_frame += 1
		#print('count_frame: ', count_frame)
		# after all the frames have been read, no point to continue;
		if(count_frame > total_frames-1):
			break
	cap.release()
	cv2.destroyAllWindows()
	print('track: ', track)

# test driver;
if __name__ == '__main__':
	# test - 01
	#filename = "C:\\Users\\yongw4\\Desktop\\JSON\\" + 'dummy.avi'
    #fps = 30.0
	#record_video(filename,  signtime = 3)
	

	# test - 02
	list_path = "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\"+ 'future_prediction.txt'
	with open(list_path, "rb") as fp:   
		info_list = pickle.load(fp)
	filename = "C:\\Users\\yongw4\\Desktop\\OP_VIDEOS\\" +  "result.avi"
	annotate_video(filename, info_list)





	