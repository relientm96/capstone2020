### Features
* Uses Human Pose Estimation as feature extraction for human gesture recognition.
* Developed and deployed our model with a proof-of-concept system as both as a desktop and web application.
* System recognises four unique human gestures with a delay of about 1-2 seconds.
* Model achieved an average accuracy of 89% using K-Fold Cross Validation.
* Our system currently **only** recognises the following four human gestures:

<div style="text-align:center">
    <div style="display:inline-block;">
        <img style="height:150px;width:auto"  src="images/Features/ambulance_sign.gif">
        <p> *Ambulance </p>
        <img style="height:150px;width:auto"  src="images/Features/pain_input.gif">
        <p> *Pain </p>
    </div>
    <div style="display:inline-block;">
        <img style="height:150px;width:auto" src="images/Features/help_sign.gif">
        <p> *Help </p>
        <img style="height:150px;width:auto"  src="images/Features/hospital.gif">
        <p> *Hospital </p>
    </div>
</div>

**These human gestures are inspired from [SignBank](http://www.auslan.org.au/) - an online resource for Australian Sign Language (Auslan).*

#### Our System
The following diagram explains a high level view of our system flow:
<br>
<div style="text-align:center">
    <img src="images/Features/system_new.png">
    <p>Figure - High Level view of our system. </p>
</div>

#### Flow of our system
1. Users would interact with a live video input from an RGB camera on their laptop.
2. The live video feed is then transmitted to specialised human pose estimation software.
3. This pose estimation software converts a video into x,y coordinates of human joints.
4. Our self-developed gesture language recognition model accepts these x,y coordinates and performs classification, outputting the word and confidence level of that sign.
5. Both the word and the confidence level output is sent to a web interface, where the user can then view.
