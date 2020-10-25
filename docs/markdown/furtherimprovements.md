### Further Improvements

##### Reducing Number of Frames:
* When using 75 frames for gesture recognition, we noticed a significant delay in model classification (taking up to 2-3 seconds).
* We wanted to explore the possibility of using less frames in our model as:
  * Recall that a human gesture class is a combination of multiple sequential gestures.
  * Eg: The gesture Pain starts with two hands fanned out, moving from shoulder height to chest height and repeats.
  * We hypothesized that in doing these gestures, humans have a fundamental limit to the rate of doing these gestures.
  * Thus, we hypothesized that we can have same recognition ability using half the number of frames (75 to 35 frames).
* We decided to cut the number of frames from 75 to 35 when doing classification.
  * Using both splits to creates two separate "training data" for that gesture class. 
* Reducing the number of frames, decreased our model's size and complexity - reducing our training time to about 40% and model recognition delay to about 1-2 seconds. 

##### Random Subframe Sampling
* Sang-Ki Ko et.al (2019) suggested the use of sampling frames from video data randomly [in this paper](https://arxiv.org/pdf/1811.11436.pdf) to improve model's generalisation performance.
* We used random subframe sampling in generating variations in our test dataset when evaluating our model.
* This simulates inconsistencies in webcam frames that may occur in online deployment (particularly for the web application).