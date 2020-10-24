### Overview

#### Human Pose Estimation

As humans, we influence the world through our bodies. We express our emotions and actions through body posture and orientation. For computers to be partners with humans, they have to perceive us and understand our behavior - recognising our facial expressions, our gestures and our movements.

Human Pose Estimation is a field of research in developing robust algorithms and expressive representations that can capture human pose and motion. Using Human Pose Estimation, a computer is able to detect human key-points from a given image, as a list of key-points such as coordinates for the person's nose or eyes. The application of human pose estimation has been extended to many fields such as generating 3D animations, medical technology and even sports.

<br>
<div class="row center-align">
    <div class="col s12 m6 l6">
        <img src="images/Overview/ai_bball_cropped.gif" style="width: 300px; height: auto">
        <p> Figure: AI Basketball Analysis.</p>
    </div>
    <div class="col s12 m6 l6">
        <img src="images/Overview/openpose_animate.gif" style="width: 300px; height: auto">
        <p> Figure: Rendering key-points onto 3D animated characters.</p>
    </div>
</div>

#### Project Objectives

Exploring the possibility of building a technology that bridges the gap between human and computer interaction, can we leverage the power of Human Pose Estimation for Gesture Recognition?

This is the goal of our project:

> Explore Gesture Language Recognition using Human Pose Estimation

To show this, our group has been working towards developing a **proof-of-concept** system that shows the effectiveness of using **Human Pose Estimation** in performing **Gesture Language Recognition**. Check out the <a href="#features">features</a> section for a high level description of our project.

#### Introduction

Recently, there have been advancements in using deep-learning techniques to perform human gesture recognition. These models are now able to take in raw videos as input, and output its guess on what that gesture is. 

<div class="row center-align">
    <div class="col s12 m12 l6">
        <img style="width:350px;height:auto" src="images/Overview/har_run.gif">
        <p> Figure: Using Deep learning to recognise human actions from a camera feed (Pose-driven Human Action Recognition and Anomaly Detection, 2019). </p> 
    </div>
    <div class="col s12 m12 l6">
        <img style="width:350px;height:auto" src="images/Overview/hand_gesture.gif">
        <p> Figure: Using Deep Learning for hand gesture recognition (Gesture Control Your Media Player with Python , 2018). </p> 
    </div>
</div>

These deep-learning based approaches fundamentally rely on the ability to extract meaningful representations of human gestures. Data representation used in training these models must incoporate temporal features to help models map a sequence of actions to a classified human gesture. The use of Recurrent Neural Networks (RNN) are commonly used in designing these models due to it's ability to understand sequential information from the past through recurrence. 

To extract sequential human features for gesture recognition, two common methods exist:
* Sensor based approaches
* Vision based approaches

Using sensor based approaches, hardware devices such as glove sensors or accelerometers are attached to humans in order to record changes in data when performing a gesture. Then, output data from these devices create temporal representations which are then used for model gesture classification training.

However, these devices need to be constantly strapped on for recognition, which makes the process of extracting human features when gesturing cumbersome. Moreover, sensor data may succumb to inaccuracies in recorded data from issues such as misplacement of sensors, hardware failure or signal attenuation. Furthermore, these devices require development costs to be produced - further making this approach neither cost efficient nor portable. Lastly, using this approach makes it difficult to extract spatial information of humans - which is important in human gesture recognition.

An alternative is to perform a vision based approach where input data takes the form of only a video, broken down into a series of sequential frames. Computer vision based deep learning models then read these ordered frames to extract both spatial and temporal information from given pixel data. These models utilise Convolutional Neural Networks (CNN), which uses image filters with trained weights applied onto images to extract spatial features.
<div class="center-align">
    <img style="width:350px;height:auto" src="https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67">
    <p> Figure: Animation of how CNNs extract features from images using a filter with trained weights (blue)
    (Convolutional Neural Networks Cheatsheet, 2019). </p> 
</div>

Using the vision based approach for human feature extraction enables portability and cost-reduction when building a solution for human gesture recognition. Through this approach, sequential information of a series of performed gestures can be extracted only using a camera and deep learning models on a computer. However, using raw images incoporates both spatial and temporal features in data for training, which increases the complexity of the model's architecture. This ultimately makes training computationally expensive and slows down model predictions - making it difficult for real time gesture recognition.
 
To circumvent this, we propose the usage of **Human Pose Estimation** software that transforms a stream of images into a stream of key-points. These key-points are numerical x,y coordinates representing estimated human body parts given an image. This significantly reduces input features for model training - moving from approximately 50,000 pixels (given 240x240 input frame resolution) to 100 key-points localising human body parts for a **single** frame. 

In our project, we have created a proof-of-concept system that first uses a camera to read in a series of frames, convert each frame to a numerical coordinate list of human body key-points and sending it to our recurrent neural network model for gesture recognition.
Our system is deployed as a web application and achieves recognition delay of about 1-2 seconds **for only four** unique human gestures.

<br>
<div class="center-align">
    <div class="row">
        <div class="col s6 m6 l6">
            <br><br><br>
            <img style="width:330px;height:auto;" src="images/Overview/matt_demo_op.gif">
        </div>
        <div class="col s6 m6 l6">
            <img style="width:250px;height:350px;" src="images/Overview/moving_keypoints.gif">
        </div>
    </div>
    <p> Figure: Using Human Pose Estimation software for landmark detection of human body key-points. Data per frame now represented as numerical coordinates instead of raw pixel data. </p> 
</div>













