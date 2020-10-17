# Capstone meeting `#4` agenda

## Agenda items
-   Progress report (11:00 - 11:10)
-   Real-time issue of pose estimation (11:10 - 11:35)
-   Reiteration of timeline or even project goal (if necessary) (11:35 - 11:50)
-   Plans for next week (11:50 - 12:00)

## Progress report
### Yick
#### Challenges
##### Limitation of OpenPose
* Cannot recognise gesture with hand only
* Cannot recognise gestures with different lighting
* Depth from the camera is also a factor in OpenPose
* Clothing may affect recognition
##### Formulation of Gesture Recognition
* Not sure if we need data sets for gesture recognition

### Matthew
- Installed CUDA and CUDNN (I have NVIDIA GPU)
- Able to get up to binaries built, couldn't run due to conflicts in environment
- Tried in Anaconda, also unable to get it to work
- [Demo on using WebCam on Google Collab](https://colab.research.google.com/notebooks/snippets/advanced_outputs.ipynb#scrollTo=SucxddsPhOmj)

### Tsz Kiu
-   Installed OpenPose on my mac
-   Able to run OpenPose on my mac, but took 20-30 seconds per frame at first
-   Tried to run O

## Possile Directions (Rabbit Holes/Focus)
* Since running OpenPose is not PC (hardware) agnostics, we focus on realizing OpenPose on a common platform which can be hardware or cloud
* So, what hardware would we use?
    * Using GPUs (expensive)
        * Can use OpenPose almost out of the box (maybe)
        * GPU is expensive
    * Using FPGAs 
        * Can relate back to our electrical engineering degree
        * Performance could exceed GPU if implemented successfully
        * How to access the FPGA
    * Using an online PC/cloud (internet only)
        * [Nectar Research Cloud](https://nectar.org.au/)
        * Google Collab / Azure Notebooks
        * Good as we can all access to it from any device
        * Possible on getting just points and feedback onto our PC
        * Make a web application to access it 
        * Concern:
            * Limitation of processing power
            * If machine dies, it'll be hard to redo
            * Real-time
            * Cost
            * Takes time to get hands on the computing platform
        * Matthew is experienced building a web app
            
* Human Gesture Detection
    * Do we need training sets? (AI approach) 
    * Using hardcoded method (mapping of points)

## Plans for next week
- Explore more on openpose libraries and demos
-  (feasbility research) Do research on different computing (cloud & hardware) options
-   Discuss with Jonathan about different computing options
    -   GPU
    -   FPGA
    -   cloud computing
-   Based on that, reiterate our timeline