### Overview

#### Australian Sign Language (Auslan)

Approximately 20,000 Australians rely on Australian Sign Language (Auslan) to communicate every day. Yet, we still see problems in communication due to the large communication gap between Auslan users and non-Auslan users. This problem becomes worse when both parties are involved in emergency like situations. 

#### Human Pose Estimation

As humans, we influence the world through our bodies. We express our emotions and actions through body posture and orientation. For computers to be partners with humans, they have to perceive us and understand our behavior - recognising our facial expressions, our gestures and our movements.

Human Pose Estimation is a field of research in developing robust algorithms and expressive representations that can capture human pose and motion. Using Human Pose Estimation, a computer is able to detect human key-points from a given image, as a list of key-points such as coordinates for the person's nose or eyes. The application of human pose estimation has been extended to many fields such as generating 3D animations, medical technology and even sports.

<br>
<div class="row center-align">
    <div class="col s12 m6 l6">
        <img src="images/gifs/ai_bball_cropped.gif" style="width: 300px; height: auto">
        <p> Figure 1: AI Basketball Analysis.</p>
    </div>
    <div class="col s12 m6 l6">
        <img src="images/gifs/OpenPoseDemo/openpose_animate.gif" style="width: 300px; height: auto">
        <p> Figure 2: Rendering key-points onto 3D animated characters.</p>
    </div>
</div>

#### Project Objectives

Exploring the possibility of building a technology that may be of use for sign language users in the future (particularly in emergencies), can we leverage the power of Human Pose Estimation for Sign Language Recognition?

This is the goal of our project:

> Exploring Sign Language Recognition using Human Pose Estimation

To show this, our group has been working towards developing a proof-of-concept system that shows the effectiveness of using **Human Pose Estimation** in performing **Australian Sign Language Recognition** in real time. Check out the <a href="#features">features</a> section for more information.

#### Related Work

Recently, there has been advancements in using deep learning techniques to perform sign language recognition based on time series classification - which is that given some sequential representation of data recorded from a human signing, classify the most likely presented sign. 

However, these deep-learning based approaches fundamentally rely on the ability to extract meaningful representations of human gestures in a sequential order to train the neural network model.

To extract human features, there exists two common methods:
* Sensor based approaches
* Vision based approaches

Using sensor based approaches, hardware devices such as glove sensors or accelerometers are attached to humans in order to record changes in data when performing a sign. However, development of these specific devices or sensors as well as their added cost, contributes to the problem of making a scalable and cost-effective solution for sign language recognition.

An alternative, is to perform a vision based approach where input data is a video, broken down into a series of sequential frames. Computer vision based deep learning models then read these ordered frames to then extract relevant features needed from the given pixel data. These models utilise Convolutional Neural Networks, which uses trained weights on filters which are applied onto image pixel data to extract features such as edges and colors.
<div class="center-align">
    <img style="width:350px;height:auto" src="https://stanford.edu/~shervine/teaching/cs-230/illustrations/convolution-layer-a.png?1c517e00cb8d709baf32fc3d39ebae67">
    <p> Figure: Animation of how CNNs extract features from images using a trained filter (blue)
    (Convolutional Neural Networks Cheatsheet, 2019). </p> 
</div>

Using vision based approaches and deep learning models enable scalability and cost-reduction when building a potential solution for sign language recognition. However, using raw image data as extracted features for model training presents the issue of having large input features, resulting in a larger model, leading to slower training time and higher model complexity.

To circumvent this, we propose the usage of Human Pose Estimation software that transforms a stream of images into a stream of key-points. These key-points are x,y coordinates representing human body and hand parts given a video. This significantly reduces input features for model training - moving from approximately 50,000 pixels (given 240x240 input frame resolution) to 100 key-points localising human body parts for a **single** frame.

<br>
<div class="center-align">
    <div class="row">
        <div class="col s6 m6 l6">
            <br><br><br>
            <img style="width:330px;height:auto;" src="images/matt_demo_op.gif">
        </div>
        <div class="col s6 m6 l6">
            <img style="width:250px;height:350px;" src="images/moving_keypoints.gif">
        </div>
    </div>
    <p> Figure: Using Human Pose Estimation software for landmark detection of human body key-points. Data per frame now represented as numerical coordinates instead of raw pixel data. </p> 
</div>












