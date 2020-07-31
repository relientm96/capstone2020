import cv2
import time

# src = https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# src - https://raspberrypi.stackexchange.com/questions/66976/capture-video-for-a-certain-time-then-quit-and-save-to-a-folder-using-opencv-3
# src - https://www.geeksforgeeks.org/python-opencv-write-text-on-video/

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
	fps = 30    # how many frames you want to write in a second;
	print('fps: ', fps)
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


	
# test driver;
if __name__ == '__main__':
	filename = "C:\\Users\\yongw4\\Desktop\\JSON\\" + 'video_hello.avi'
	#fps = 30.0
	record_video(filename, fps = 30, signtime = 3)

	