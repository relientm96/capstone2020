# Capstone Meeting #9 -- 31/4/2020

## Agenda
* Check GRA before submitting; [Done]
* Each person update on previous week's work (briefly!)
* Add/Update our WBS for AusLan
* Go in depth on insights gain from sign language recognition research
* Discuss about our engagement with AusLan community? (Not priority for now?)

### Research Updates
#### Yong Yick
* Researced on LSTM: Long short term memory
* LSTM is a specific implementation of RNN that 
* Introductory Explanation: [Explanation](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* OpenSource and can be built on Pose Estimation
* Will get a list of papers so that we can refer to it.
* Look at other research topics that may be similar.
* Managed to narrow down to a few research papers that have sufficiently large dataset for AusLan.
* Need for annotated datasets.
 
#### Tsz Kiu
* [Possible Dataset](https://github.com/Signbank/Auslan-signbank/tree/master/signbank)
* Learnt alphabets for Auslan.
* Learnt that fingerspelling is quite dynamic in movements as well.
* Looking at Auslan translation based on computer vision.
* Steps that they took:
    * (Segmentation) - only face; only hand; or both?;
    * (Feature Extraction) - 
        * Used to train the model
        * Further process such as angle of joints; etc...
    * (Gesture Recognition) -
        * Inputting keypoints + features extracted into our model to predict sign + word.
* Segementation to get the necessary body parts.
* We need to narrow down a list of features that we want to extract that would be useful for our case.
* Gesture Recognition research --> Hidden Markov Model.
* [Implementing Hidden Markov Model](http://htk.eng.cam.ac.uk/)
    * HTK consists of a set of library modules and tools available in C source form. The tools provide sophisticated facilities for speech analysis, HMM training, testing and results analysis. 
* Problem with segementation/feature extraction without Pose Estimation Keypoints:
    * Only able to detect the locations of joints
    * Can't work with dynamic signs and movements

### Phases in our Project
#### Phase 1 -- Static Gestures
* Static Alphabets / Numbers
* Sign Language for one word and static gesture (Use Australian Wide version)
* Set a dictionary a set number of words/alphabets
* Collecting training datasets;
    * Annotated images or videos
    * Possibly re-process raw videos before feeding into models
    * Possibly look into asking UniMelb department to get Sign language datasets.
* Look into implementing with different models:
    * CNN
    * LSTM
    * Can look into image classification algorithms
    * etc...
* Collecting test datasets and testing.

#### Phase 2 -- Moving Gestures
* Temporal Alphabets (J and H)
* Detecing words with temporal gestures
* Workflow similar to phase 1

#### Phase 3 -- Series of Gestures
* Detecting a sequence of alphabets to form a word
* Detecting sequence of words
* (POSSIBLE WAY): Taking a series of words and using another AI/BOT to rearrange words in a grammatically correct way
* Workflow similar to phase 1

### Concerns
* Gesture Recognition seems to be quite a challenging task due to time constraints.
* Grammar can be an issue when we focus only on finger spelling.

### Moving forward 

#### Administrative / Assignments
* [Yick] Submit our GRA by 1st May -- Yick to Submit
* [Yick] Project brief - auslan (not urgent) - Latex;
* [Yick] template for final project;
* [Matthew] Working with IP Camera Stream
* Assignment 2?

#### Capstone
* [Yick] set up and run different models (primarily LSTM);
* (team) let's set up and run different models (in terms of generalization, accuracy, ease of use);
* (Team) Implicitly, continuing researching the techniques/rabbit hole (also as self-development)
* (Team) let's find out different models (not urgent);
* Liaise with AusLan communities to get datasets 
(in a few weeks)
    * [Matthew] - Unimelb subject/departments
    * [yick] - auslan, other universities;
* Start colllecting possible datasets from sources.
(in two weeks)
    * Using web scrapping libraries
    * Manually downloading for personal testing;

#### Good-to-do
* approach lecturers (jingge, erik, jonathan, etc.);