### Application Deployment

<div class="center-align">
    <img style="width:500px; height:auto;" src="images/SystemApplication/Demo_Gif.gif">
    <p> Figure: Demonstration as a desktop application. </p>
</div>

* A huge part of our project was dedicated toward creating a system that deploys our model for it to recognise gestures using a live camera feed.
* We've created two application variants:
  * Desktop Application - Uses OpenCV Python, and OpenPose Python
  * Web Application - Uses WebRTC, Socket.io, aiohttp, aiortc and Tensorflow.js

##### Hardware Requirements
* In order for OpenPose to perform real-time human pose estimation, it requires the following hardware requirements:
  * Requires the use of CUDA 10.0 Nvidia GPU Computing Toolkit, with CUDNN = 7.5
  * Nvidia Graphics Processing Unit (GPU) of about 8 GB RAM.
* Thus to run our application, we decided to develop and deploy our system using two Virtual machines provided by the Melbourne School of Engineering (MSE) IT Department.

##### Rolling Window

<div class="center-align">
  <img width="800px", height="auto" src="images/SystemApplication/rollingwindow.png">
  <p> Figure: Illustration of how the rolling window works - acting as a shift register of key-points </p>
</div>

* Unlike offline predictions, our application requires specialised data-structures that can store live key-points for online prediction.
* Thus, we decided to use the **rolling window** data structure, which acts as a shift register of key-points.
* This data structure is defined as as python class:
  
```python
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
	
	def addPoint(self, arr):
		# Check for same number of joints in input array before we insert
		if ( len(arr) != self.numbJoints ):
			print("Error! Number of items not equal to numbJoints = ", self.numbJoints)
			return False

        # shift register;
        # remove the oldest from the first index;
        # add the most recent entry always enters from the "last" index 
        self.points = np.delete(self.points, 0, 0)
        self.points = np.vstack([self.points, arr])

		return True

	def resetWindow(self):
		'''
		Reset keypoints to duplicate all across all indexes
		'''
		self.isReset = True
```
 
* The key operations in this rolling window data structure is:
  1. Initialize the rolling window, by filling 35 key-point lists with zero values, each key-point list having 98 elements.
  2. First, the rolling window receives a 98 sized key-point list (generated from a single frame)  
  3. The oldest key-point in the rolling window is removed.
  4. The new incoming key-point list is added to the rolling window.
  5. Repeat steps 2,3 whenever a new key-point comes in again.

##### Workflow
* In both applications, they rely on a fundamental workflow:
  * Camera connects and streams frames to the OpenPose program.
  * The OpenPose program converts the video as a series of frames into a series of key-point lists.
  * We then used the rolling window data structure to append this new key-point list to the window (and remove oldest key-point list).
  * Then, we send the 35 length rolling window to our model for classification.

#### Desktop Application
* Our desktop application uses OpenCV-Python an OpenPose Python.
* The main workflow of the desktop application is simple, following exactly the same as the workflow listed above.
* For more details, refer to the (source code here)[https://github.com/relientm96/capstone2020/tree/master/src/desktop]

#### Web Application
* Creating a web application is a much harder system.
* Our Web Application uses:
  * WebRTC - Web Real Time Communication, used to handle live streaming of camera frames from user's browser to server.
  * AioHttp,AioRTC - Python async-io libraries for our webserver to receive frames and run OpenPose.
  * Socket.io - Web Socket Library, for full duplex connection between client and server (avoids request-response pattern).
  * Tensorflow.js - Our gesture recognition model trained in keras, is converted into a Tensorflow.js model for deployment, sitting on the user's browser.
* We know that for web-server, gaining access to the camera, and transmitting frames over the internet is a complex issue.
* We have two other older iterations of our web application, before our final version. 

##### Version 1 (First Draft)
<br>
<div style="text-align:center">
    <img src="images/SystemApplication/app_draft.png" style="width:700px; height:auto">
    <p>Figure - Version 1 of our working web application design.</p>
</div>

* Here, the browser is only responsible to connect to the user's camera.
* Then, using an interval function, it encodes these frames as base-64 encoding (representing image data as characters)
* The base-64 encoded message is then transmitted to the web-server using web sockets (via socket.io)
* On the server side, it receives and decodes the base-64 encoded frame, and passes it to OpenPose for key-point processing.
* Once the key-points are rendered, they are sent to our gesture recognition model, which is deployed as a Keras model sitting on the server.
* Once classification is complete, we transmit back both the rendered skeleton over the frame (encoded as base-64 again) and the classified word and probability for the user to view on the browser.

**Problems**
* We noticed that there was significant latency in receiving rendered human skeleton images coming from the server (rendered frames coming back at 1-1.5 fps).
* We also noticed that the server was performing significant amount of work by doing both OpenPose processing of frames and model classification, while handling message passing between server and client using web sockets.

##### Version 2 (Second Draft)
<div style="text-align:center">
    <img src="images/SystemApplication/WebAppVer2.png" style="width:800px; height:auto">
    <p>Figure - Version 2 of our working web application design.</p>
</div>

* Workflow is similar to the first draft of our system.
* Key Difference: To reduce workload on the server, we decided to port our gesture recognition model onto the browser instead.
* This uses Tensorflow.js to convert our trained Keras Model and deploy it onto the browser.
* This enables the workload of doing classification to be done on the client's side instead, rather than on the server.
* By doing this, we saw improvements - increasing frame rate of rendered skeleton frames to about 2-3 fps.

**Problems**
* However, our system still uses the method of encoding/decoding images into base-64 strings, increasing processing overhead.
* These base-64 encoded strings are also transmitted using web-sockets, which are TCP based.
* TCP based connections are reliable (using acknowledgements between server/client) which slows down transmission speeds.
  
#### Version 3 (Final)
<div style="text-align:center">
    <img src="images/SystemApplication/System_Diagram_Detail.png" style="width:800px; height:auto">
    <p>Figure - Application Workflow as a Real Time Sign Language Recognition on the web</p>
</div>

* In our newest version of our application, we utilised Web Real Time Communication (WebRTC) to handle streaming of frames to and from the server.
* WebRTC is UDP based, which is an unreliable transmission protocol achieving faster speed for real time transmission.
* With WebRTC, base-64 encoding/decoding of frames are no longer used.
* Using the final version of our application, we are able to achieve transmission of rendered skeleton frames at about 4-5 fps.
* Because of the use of WebRTC, our webserver changed from using flask to aiohttp and aiortc - asynchronous python based web server frameworks that support WebRTC.







