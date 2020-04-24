# Capstone Meeting #8 (24/04/2020)

## Agenda
| Schedule | Item |
|----------|------|
| 11:00 - 11:15 | Progress report |
| 11:15 - 11:35 | Final check on Assignment 1 Part 1 |
| 11:35 - 11:50 | Discuss things to do for Part 2 |
| 11:50 - 12:05 | WBS for Auslan |
| 12:05 - 12:20 | Tasks to do this week |
| 12:20 - 12:30 | Other concerns |

## Final check on Assignment Part 1
- Already did a final check using rubric.
- Submitting version now on LMS.

## Discuss things to do for Part 2 (or a final check)
### Team SWOT Analysis
#### Tsz Kiu
Strengths:
- Strong interest and knowledge in software
- Familiar with music 
- Keen on learning
- Interest in field of research
- Experienced with music-related research

Weakness:
- Not familiar with machine learning/gesture recognition
- Not familiar with Windows OS 

Opportunities:
- Picking up machine learning

#### Matthew
Strength:
- Collaborative tools
- Cloud computing
- Keen on learning

Weakness:
- Machine learning
- Reading research paper

Opportunities:
- Wants to pick up machine learning

#### Yick
Strength:
- Comfortable with computing/programming
- Forward Thinking
- Good at organizing workflow of doing assignment
 
Weakness:
- Lack of formal trainings/fundamentals

Opportunities:
- Learning proper techniques in machine learning

## Progress Report
- Everyone's update from previous week (before Assignment 1 stuff)
### Matthew
- IP camera: ~40fps
    - IP camera on localhost using cam2web
    - cam2web issue: camera resolution
- Webcam issue: maybe the issue is on OpenCV side
    - Not much help online
    - Need to read documentation
- UDP stream
    - What the output of pose estimation should be?
    - This depends on the gesture recognition block

### Yick
- Gesture recognition is not that straight forward
- Gesture recognition technologies now do not require pose estimation keypoints. 
- Datasets depend on what current Gesture Recognition algorithms use
- There are two parts of our project:
    - Research
        - Figure out which algorithms to employ
        - Search up on how well OpenPose works in gesture recognition
    - Implementation
- Using data to train model itself 
    - Need large datasets with annotation
    - Can be found online, just need to process
- Transfer learning
    - Using other people's models to train ours
    - But requires a lot of research in altering datasets

### Tsz Kiu
- Searching up on Music Related stuff
- Able to link json output to PureData in real time
    - Reading each json file one by one
    - Using coordinates of body parts to do stuff in PureData

## WBS for Gesture Recognition
- Talk about our steps for transition to AusLan]
- Give ourselves 1.5 weeks to gather insights while developing our WBS

## Tasks for this week
- Debugging the webcam issue (Matthew)
- Start working on the Project Charter (Team)
- Finish the WBS for Gesture Recognition in two weeks (Team)
- Finding and document the gesture recognition research built on top of body coordinates (maybe for two weeks, could be longer) (Tsz Kiu)
- Find and set up open source code on gesture recognition built on top of (ideally) OpenPose (Yick)
- Send Email to Jonathan after project charter submission (Tsz Kiu)
- GRA; due date (Yick)
- Assignment 02 (yick)

## Tasks for the future (potentially including in WBS)
- Come up with a class of gestures from Auslan to focus on (Time variant or Time invariant)
- Liaise with the deaf community

## Other concerns (if anything)
- Drafting email to Jonathan describing our project change