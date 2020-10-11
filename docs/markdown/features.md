### Features

We have successfully developed a system that is able to recognise the following Auslan signs:

<div style="text-align:center">
    <div style="display:inline-block;">
        <img style="height:150px;width:auto"  src="images/gifs/ambulance_sign.gif">
        <p> Ambulance </p>
        <img style="height:150px;width:auto"  src="images/gifs/pain_sign.gif">
        <p> Pain </p>
    </div>
    <div style="display:inline-block;">
        <img style="height:150px;width:auto" src="images/gifs/help_sign.gif">
        <p> Help </p>
        <img style="height:150px;width:auto"  src="images/gifs/hospital.gif">
        <p> Hospital </p>
    </div>
</div>

#### Our System
The following diagram explains a high level view of our system flow:
<br>
<div style="text-align:center">
    <img src="images/system_new.png">
    <p>Figure 1 - High Level view of our system. </p>
</div>

#### Flow of our system
1. Auslan users would interact with a live video input from an RGB camera on their laptop.
2. The live video feed is then transmitted to specialised human pose estimation software.
3. This pose estimation software converts a video into x,y coordinates of human joints.
4. Our self-developed sign language recognition model accepts these x,y coordinates and makes a prediction on the word that the Auslan user is doing as well as the probability of that sign.
5. Both the word and the probability output is sent to a web interface, where the user can then view the outputs.
