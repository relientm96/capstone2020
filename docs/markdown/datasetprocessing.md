### Dataset Processing

In this section, we will cover how we collected and processed raw video data to be used for our model training. 

#### Data Collection

##### Video Collection
* A major issue we encountered, was the lack of Auslan video sign data-sets available for use online.
* Thus, we resorted to:
  * Recording ourselves as video data, doing about 400 raw video signs.
  * Each sign having approximately 100 videos.
  * Each video 3 seconds in length.
* Using our webcams at 30 FPS, we have approximately 75 frames per video for model training.  

##### Video Augmentation

<div class="center-align">
    <img width=350px, height=auto, src="images/DataProcessing/ImgTransform.png">
    <p>Figure: Example of using Image Augmentation techniques to frames acting as "new data" for model </p>
</div>

* An alternative way we approached in creating more "data" for our model to train on was to perform **Classifcal Image Affine Transformation** on video frames.
* We explored the usage of these techniques to simulate "real life" situations that may occur when signing to a webcam such as:
  * Image flipping - Simulating "same sign" for left-handed and right-handed signers.
  * Image Shearing - Simulating situations where person is standing when signer is not parallel to webcam.
  * Image Rotation - Simulating situations where camera is rotated with respect to signer.
* Apart from visual augmentation, we've also explored usage of speed augmentation to videos - simulating different signing speeds for different signers.

<br>

##### Video Processing

<div class="center-align">
    <img width=750px, height=auto, src="images/DataProcessing/TranslateInTime.png">
    <p>Figure: Process of feeding a single 3 sec video for a sign into OpenPose to generate a series of keypoints </p>
</div>
<br>

* Once videos were recorded, we used OpenPose to translate these videos (a series of frames) into a series of 98 key-points.
* For a single frame, OpenPose returns each body joint keypoint in the structure of `x1,y1,c1,x2,y2,c2,...xi,yi,ci` 
  * where x_i represents the i'th joint's x-coordinate
  * y_i represents the i'th joint y-coordinate
  * c_i represents the confidence level (probability) of that joint's coordinats.
* For a single frame, a json structure is returned like:
```json
{
    "people":[
        {
            "pose_keypoints_2d":[582.349,507.866,0.845918,746.975,631.307,0.587007,...],
            "hand_left_keypoints_2d":[746.975,631.307,0.587007,615.659,617.567,0.377899,...],
            "hand_right_keypoints_2d":[617.581,472.65,0.797508,0,0,0,723.431,462.783,0.88765,...]
        }
    ]
}
```
* We then flatten and combined all 98 keypoints:
  * 8 from pose_keypoints
  * 21 from hand_left_keypoints_2d
  * 21 from hand_right_keypoints_2d
* Removed all confidence level values (removing c_i's)
* into a single array like [b]:
```
[j0_x, j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, ...j98_x j98_y]
```

##### Keypoint Augmentation
* [To Be Added]


##### Data Formatting 
* Before training, we would need to label our input/output data correctly.
* Let X be the input data to our model - a collection of 75 flattened arrays in [b]. 
* Let Y be an integer encoding of a respective class that these series of X frames represent using the following class map:
```javascript
dictOfSigns = {
    0: "ambulance",
    1: "help",
    2: "hospital",
    3: "pain",
    4: "ThumbsUp"
};
```
* We stored X,Y data in .txt files (and later numpy pickled files) in the following format:
* X Data (input) 
```
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Pain, Frame 1
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Pain, Frame 2
...
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Pain, Frame 75
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Help, Frame 1
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Help, Frame 2
...
[x1 y1, x2 y2, x3, y3, x4, y4, x5, y5 ... x98, y98]  --> For Sign Help, Frame 75
```
* Y Data (output class labels)
```
3 --> Indicates that series of 75 arrays represents the sign "pain"
1 --> Indicates that series of next 75 arrays represents the sign "help"
```



