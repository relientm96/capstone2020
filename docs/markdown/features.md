### Features

We have successfully developed a system that is able to recognise the following Auslan signs:

<div style="display:inline-flex">
    <img src="https://github.com/relientm96/capstone2020/blob/yick_vm_pipeline/meeting-logs/JonathanLogs/gifs/ambulance_sign.gif?raw=true">
    <img src="https://github.com/relientm96/capstone2020/blob/yick_vm_pipeline/meeting-logs/JonathanLogs/gifs/pain_sign.gif?raw=true">
</div>

<div style="display:inline-flex">
    <img src="https://github.com/relientm96/capstone2020/blob/yick_vm_pipeline/meeting-logs/JonathanLogs/gifs/help_sign.gif?raw=true">
    <img src="https://github.com/relientm96/capstone2020/blob/yick_vm_pipeline/meeting-logs/JonathanLogs/gifs/hospital_sign.gif?raw=true">
</div>

#### Our System
At a high level, our system can be viewed as the in the following diagram:
<br>
<div style="text-align:center">
    <img src="images/system_new.png">
    <p>Figure 1 - High Level view of our system. </p>
</div>

##### Flow of our system
1. Auslan users would interact with a live video input from an RGB camera.
2. The live video feed is then transmitted to specialised human pose estimation software.
3. This pose estimation software converts a video into x,y coordinates of human joints.
4. Our self-developed sign language recognition model accepts these x,y coordiantes and makes a prediction on the word that the Auslan user is doing as well as the probability of that sign.
5. Both the word and the probability output is sent to a Graphical User Interface, where the user can then view.
