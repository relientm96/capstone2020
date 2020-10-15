### Human Pose Estimation

#### OpenPose

* We used [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), an open source Real Time multi Human Pose Detection software.
* Developed and maintained by a group of computer science researchers from Carnegie Mellon University.
* Written in C++ and Caffe.

<div class="center-align">
  <img src="images/PoseEstimation/Zhe_Demo_OP.gif">
  <p> Figure : Demonstration of OpenPose - multi-person 2D Human Pose Estimation (From Realtime Multi-Person 2D Human Pose Estimation using Part Affinity Fields, 2016) </p>
</div>

#### OpenPose Approach

* There exists two main approaches in doing human pose estimation:
  * Top down approach  - Detect where humans are in an image, then infer the respective human keypoints.
  * Bottom up approach - Detecting human keypoints first, then inferring humans from an image. 
* OpenPose uses a **bottom-up** approach in doing Human Pose Estimation.

<div class="center-align">
  <img src="images/PoseEstimation/OpenPosePipeLine.PNG" width='800px', height='auto'>
  <p> Figure : High level view of OpenPose pipeline from Realtime Multi-Person 2D Pose
  Estimation using Part Affinity Fields.
  </p>
</div>
  
* The following is a rough break-down on how OpenPose extracts human keypoints from images, for a more in-depth explanation check out [the original paper here](https://arxiv.org/abs/1812.08008):
  1. An input RGB image is fed to a multi-stage CNN model.
  2. The mutli-stage CNN model contains two different branches.
    * Top branch (Fig a): extracts the confidence maps for body part detection such as where eyes, elbows and others are on image.
    * Bottom branch (Fig b): predicts the affinity fields - representing a degree of association between different body parts.
  3. After this, it uses bipartite matching and a greedy parsing algorithm to then make associations between body part candidates.
  4. To ultimately, form the assembled full pose output.

#### How we used OpenPose
* We utilised OpenPose to extract out the following human keypoints:
  * 7 Skeleton Pose Keypoints (points 0 to 7 from figure)
  * All 21 Keypoints from Right Hand
  * All 21 Keypoints from Left Hand
* Thus, for each frame, we use a total of **98 key-points** as our input feature to our model for sign language recognition.

<br>

<div class="center-align">
    <div class="row">
        <div class="col s12 m6 l6">
            <img style="width: auto; height:350px" src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_25.png">
            <p> Figure: OpenPose Human Pose Key-point Landmark Mappings. </p>
        </div>
        <div class="col s12 m6 l6">
            <img style="width: auto; height:350px" src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_hand.png">
            <p> Figure: OpenPose Hand Key-point Landmark  mappings. </p>
        </div>
    </div>
</div>





