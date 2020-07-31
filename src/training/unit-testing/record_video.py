import cv2
import time

# src = https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
# src - https://raspberrypi.stackexchange.com/questions/66976/capture-video-for-a-certain-time-then-quit-and-save-to-a-folder-using-opencv-3
# src - https://www.geeksforgeeks.org/python-opencv-write-text-on-video/

filename = 'video_hello.avi'
fps = 30.0

# capture frames from a camera with device index=0
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))


# Define the duration (in seconds) of the video capture here
preptime = 3
capture_duration = preptime + 3.3
# loop runs if capturing has been initialized 
#while(1): 
start_time = time.time()

while( int(time.time() - start_time) < capture_duration ):
	
	# reads frame from a camera 
	_, frame = cap.read() 
	
	# describe the type of font 
	# to be used. 
	font = cv2.FONT_HERSHEY_SIMPLEX 
  
	# insert text;
	tick = int(time.time() - start_time)
	if(tick <= preptime):
		TEXT = "Start signing in " + str(tick)
		cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4)
		cv2.imshow('Camera',frame) 
	else:
		start_recordtime = time.time()
		TEXT = "OOO, Handsome! " + "tick: " + str(tick-preptime)
		cv2.putText(frame, TEXT, (50, 50),  font, 1,  (0, 255, 255), 2, cv2.LINE_4) 
 
		# Display the frame
		cv2.imshow('Camera',frame) 
		out.write(frame)

	# user could force shut down by pressing "Q"
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# release the camera from video capture
cap.release() 

# De-allocate any associated memory usage 
cv2.destroyAllWindows() 