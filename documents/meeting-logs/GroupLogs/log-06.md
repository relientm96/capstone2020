# Capstone Meeting #6 
Date - 10/4/2020
## Agenda
1. Go through GitHub boards to check issues/tasks done from last week.  
2. Talk about assignment 01 details.  
    - Gantt chart
        - How do guys want to draw it?
        - Github can assign due dates to milestones, but not to projects.
        - Github API can check which milestone an issue is linked to, but can't track which project it is linked to
        - Whether we are automating it or not, we still have to have a clear idea of what we want to draw.
3. Issues with Horizon
    - Reply to Lucas email
4. Discuss project parts, namely:
    - OpenPose part 
        - Demo On Google Colab
        - Demo on Matthew's PC
        - Demo on VM
        - Issues faced
        - Work arounds
    - Gesture Recognition part
        - Research done 
        - Implementation (AI/Signal Processing)
        - Platform + Tools? (Programming Language?)
5. Finalize tasks for coming week.

## Meeting Notes

### Assignment 01 Details

#### Gantt Chart
* Gantt Chart Generation
    * Need to deal with authentication issue
    * Tsz Kiu has created a skeleton/structure
    * [TODO] We need to create a list of milestones and tasks that we want to group together
    * [TODO] Someone can set up a template (on google sheets) 

#### Assignment 01 Document
* We have to stick to the structure using docx.
* Two docouments; project-charter & general risk assestment

### OpenPose 
* We have a working demo on Matthew's PC
* Output can be in terms of json files for each frame or a video output
* Matthew is working on creating json output files to aid in gesture recognition system development

### Gesture Mapping
* Breaking down into two sections
    * [for starters] One to one mapping from json to Gesture/music program
        * could be others: region mapping (Matthew's idea)
        * high level gesture
        * low level gesture
    * Implementation based on livestreaming approach
    * be creative for the music program, imagination
    * research the gesture-recognition using AI;
        * feasibility?

## VM Horizon
* Given access by Lucas to access VM concurrently.

### Issues
- CUDA Device not recognized on VM 
- Windows Users (Matthew + Yick) not able to link audio and video from VM
- Root permission issues?

## ToDo
1. Create a document outlining our revised project timeline/milestones for gantt chart - Matthew 
2. Set up a template for gantt chart - Yick
3. Set up a structure for our assignment 1 charter - Yick
4. Uploading OpenPose datasets (output - json) on GitHub - Matthew
5. exploring the APIs from OpenPose and Pure Data / SuperCollider - Tsz Kiu and team
6. Setting up a basic mapping system from OpenPose outputs. - team
7. Block diagram of our system? (Tsz Kiu)
8. VM Nvidia Driver Installations - Matthew
9. Emailing Jonathan Progress - Tsz Kiu
10. Emailing Lucas Issues that we are still having (Monday) - Tsz Kiu
 