# Real-time Gesture Mapping with Pose Estimation

## Criteria to focus on
### Background problem
Although gesture control have became a trend since this century, many of them involves wearables.
(__TODO__: we might need a stronger motivation?)

Here are some examples of gesture control in music:
-   [MyoSpat](https://www.researchgate.net/publication/320427332_MyoSpat_A_hand-gesture_controlled_system_for_sound_and_light_projections_manipulation)
-   [SoundMorpheus](https://www.researchgate.net/publication/327075959_SoundMorpheus_A_Myoelectric-Sensor_Based_Interface_for_Sound_Spatialization_and_Shaping)
-   [gSPAT](https://www.researchgate.net/publication/280009422_gSPAT_Live_sound_spatialisation_using_gestural_control) (Search this word with care please)
    -   [video](https://www.youtube.com/watch?v=CBtKvhNAcCQ)

### Goal
To give more freedom to the user and make it more accessible, we propose to build a system where users are able to control musical parameters without any wearables but just webcam(s).
### Project description and idea of tasks
-   Identify a pose/gesture recognition system that would be able to do pose estimation in real-time ([OpenCV](https://github.com/opencv/opencv), [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), etc.)
-   More about gesture recognition in music (in real-time)
    -   [Real-time Gesture Mapping in Pd Environment using Neural Networks](https://pdfs.semanticscholar.org/12fd/324108003ab23a41d28871772c4437836cb5.pdf)
-   Identify ways to interact with computer in a musically meaningful way (we can refer to the papers above)
-   Mapping strategies (have a look at [Tanaka](https://research.gold.ac.uk/6834/1/P88_Tanaka.pdf) and [Wanderley](https://www.researchgate.net/publication/2765549_Instrumental_Gestural_Mapping_Strategies_as_Expressivity_Determinants_in_Computer_Music_Performance)
-   Hook the output (gestures recognized) of the above system to an open source digital music software ([Pure-data](https://github.com/pure-data/pure-data), [SuperCollider](https://github.com/supercollider/supercollider), etc.)

### Reasons to backup
-   Not much at the moment apart from that music interests me in general, can lookup for more reasons/motivations if you guys are onboard.

## Ideas from Cont et al. (2004)
### Paper
[The paper is here](https://pdfs.semanticscholar.org/12fd/324108003ab23a41d28871772c4437836cb5.pdf)
### Background problem
>   Gesture control of musical events have become a trend in computer music over the past years.
>   On the other hand, most sensor mappings approaches in this community are fixed and confined to few parameters thereby do not allow much control and freedom over musical events.
### Reasons to backup
-   Though dealing with neural network, does not require much expertise to train and maintaining the network.

## Ideas from Di Donato et al. (2015)
### Paper
[The paper is here](https://www.researchgate.net/publication/280009422_gSPAT_Live_sound_spatialisation_using_gestural_control)
### Background problem
Although there are many tools for gesture control in music, normal users may not always have the knowledge to setup and use these powerful tools.

## Other ideas with gesture control in Music
-   Computer gesture to translate sign language to english?
    -   Gesture control and real-time speech synthesizer
