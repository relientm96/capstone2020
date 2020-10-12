### Application Deployment

[Work in Progress]

#### Desktop Application
* We have created desktop version of our application, using python's OpenCV library.

<div class="center-align">
    <img style="width:500px; height:auto;" src="images/SystemApplication/Demo_Gif.gif">
    <p> Figure: Demonstration as a desktop application. </p>
</div>

#### Workflow
* The OpenCV program reads frame by frame from the computer's webcam using `cap.read`.
* Datum object is a custom defined OpenPose data structure - that stores keypoints in `numpy arrays`.
* Datum gets passed to our gesture recognition module, that receives an input keypoints.
* Using a RollingWindow, a queue like structure that performs both Enqueue and Dequeue when a keypoint comes.

#### Web Application

A huge part in our project was dedicated towards building an online real-time web application for users to experience real-time Auslan recognition using our model.

##### Version 1
<br>
<div style="text-align:center">
    <img class="materialboxed" src="images/SystemApplication/app_draft.png" style="width:700px; height:auto">
    <p>Figure - Version 1 of our working web application design.</p>
</div>

##### Version 2
<div style="text-align:center">
    <img src="images/SystemApplication/WebAppVer2.png" style="width:800px; height:auto">
    <p>Figure - Version 2 of our working web application design.</p>
</div>

#### Version 3 (Final)
<div style="text-align:center">
    <img class="materialboxed" src="images/SystemApplication/System_Diagram_Detail.png" style="width:800px; height:auto">
    <p>Figure - Application Workflow as a Real Time Sign Language Recognition on the web</p>
</div>





