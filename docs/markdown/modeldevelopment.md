### Model Development

##### Problem Formulation
* From the Human Pose Estimation & Dataset Processing sections, we've covered the methods, processes and tools used when doing **feature extraction** in order to train our model to classify signs.
* Recall that we formulate the problem of gesture recognition as time-series classification where:
  * Given a series of sequential N data points in time
  * Train a model to classify specific class for that given sequence of N data points.
* Similarly, looking into gesture recognition in our case:
  * We are given a series of sequential 75 key-point lists (each list having 98 human key-points).
  * Train the model to **classify** particular gesture given 75 key-point lists.
* As mentioned earlier in the Background section, our model heavily relies on Long Short Term Memory Networks (LSTMs) in order to handle sequential order of key-points for gesture classification.

##### Our model 
<div class="center-align">
  <img width="450px", height="auto", src="images/Model/model.png">
  <p> Figure: Our gesture recognition model architecture. </p>
</div>

* When feeding our model for training, we shape our input according to the dimensions of (x, 75, 98) where:
  * x - Batch size used during training.
  * 75 - Number of keypoint lists (converted from frames) representing this gesture.
  * 98 - Each keypoint list has a 98 human key-points.
* Our model uses two LSTM layers each with 64 hidden units.
* Each LSTM layer is then applied **dropout**, at a rate of 0.5 (50%).

 <div class="center-align">
    <img width="500px", height="auto", src="https://miro.medium.com/max/875/1*iWQzxhVlvadk6VAJjsgXgg.png">
    <p>Figure: Comparison between a normal neural network and one that applies dropouts (Dropout in (Deep) Machine learning, 2016) </p>
  </div>

  * Dropout is a regularization technique used in preventing a model to overfit on a small set of training data. 
  * Given a ratio (0-1), it randomly drops a ratio of the layer output nodes.
  * This prevents units from co-adapting too much and prevent the model from overfiting on a small dataset.
  * The usage of dropouts was inspired by the following [paper](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf).
* Final layer is a **softmax** dense layer, that outputs scores to the corresponding four classes. 
  
  <div class="center-align">
    <img width="300px", height="auto", src="images/Model/softmax.png">
    <p>Figure: Illustration of how the softmax function maps a list of inputs to outputs in a range between (0,1). </p>
  </div>

  * Using the above function, each class is assigned a "score" that is between the range of (0,1).
  * The sum of all outputs of the softmax layer, add up to 1.
  * The higher the value, the more confident the model is in classifying that particular gesture.
* We chose the Adam Optimizer as our model's learning algorithm.
* We chose the sparse categorical cross entropy loss function - used in quantifying classification errors.

##### Model Development Process

<div class="center-align">
  <img width="600px", height="auto", src="images/Model/model_dev_process.png">
  <p> Figure: Diagram workflow when developing our model. </p>
</div>

**First Phase**
* We first split our collected dataset using a random integer split, where
  * 80% of the dataset - Training Data
  * 20% of the dataset - Test data
* Then, we chose some set of initial parameters to use in creating our "first-draft" model of our system.
* Key choices made for our first draft model:
  * Using LSTMs as core for sequential classification of gestures.
  * Using Dropouts at 0.5 after each LSTM layer.
  * Using a Softmax as output layer to score confidence levels for each class.

**Second Phase**
* After a set of parameters are chosen, we are

##### Hyperparameter tuning:
  * Hyperparameter tuning is an optimization problem of choosing the best set of model parameters for a deep learning model using a dataset.
  * We decided to optimise our model by running hyperparameter tuning on the following model parameters:
    * Number of LSTM Layers
    * Number of Hidden Units in LSTM Layer
    * Dropout to apply for that LSTM layer
    * Number of Epochs
    * Batch Size
  * To optimize, we defined the following search space for the parameters listed above:
  ```
  search_space = {
        'Number of LSTM layers': [1, 2, 3]
        'Number of Hidden Units in LSTM layer': [32, 64, 128, 256, 512,1024]
        'Dropout to apply after each LSTM layer': [0.1,0.2,0.3,0.4,0.5] 
        'Batch Size' : [32, 64, 128],
        'Number of Epochs' : [50, 70, 80, 100] 
  }
  ```  
  * We used the [hyperopt framework](https://github.com/hyperopt/hyperopt) - a python library that enables the use of bayesian optimization for hyperparameter tuning. 
  * To use Hyperopt, we defined
    * Our search space as list of values as listed previously.
    * An objective function to minimize `f_min()` - in our case, we used the **negative** validation accuracy as the scalar-valued output to be minimized (in turn maximizing validation accuracy).
  * We decided to not tune with different model optimizers as using other optimizers did not largely impact our model performance.
  * We also used [Apache Spark](https://spark.apache.org/docs/3.0.1/) to further increase data processing speed in hyperparameter tuning.
