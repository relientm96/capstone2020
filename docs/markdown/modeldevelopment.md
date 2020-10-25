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
* We chose the cross entropy loss function to quantify our model's loss for each output classification.
  * Cross Entropy Loss: \\[ -\sum_{j=1}^{M} t_{j} log(y_{k}) \\] 
  * Where 
    * \\(M\\) is the number of classes (in our case M = 4 gestures)
    * \\(t_{k}\\) ground truth value for class \\(j\\), (outputs 1 for class j, 0 otherwise)
    * \\(y_{k}\\) softmax output confidence level value (between 0-1)
  * Using this allow us to quantify smaller values in \\(y_{k}\\) as a larger error made and vice versa.
  * Eg: model attempts to predict a confidence level for output labeled "pain", 
    * Further from ground truth, larger error for backpropogation : \\(y_{k} = 0.3\\) , \\(-log(0.3) = 1.2 \\)  
    * Nearer to ground truth, smaller error for backpropogation: \\(y_{k} = 0.9\\), \\(-log(0.9) = 0.105\\)
  * As noted by the [Tensorflow API](https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy), we used sparse categorical cross entropy as our output labels are integer encodings.

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
* Our initial model gave us a rough starting point to deploy in our system to test for effectiveness. 

**Second Phase**
* In the next phase, we then consulted research papers and performed hyperparameter tuning to refine our model parameters.
* Suggested by [Reimers (2017)](https://arxiv.org/pdf/1707.06799.pdf), possible optimal hyperparameters for LSTM models are:
  * Optimizer - parameter choice has high impact, suggested the use of Adam or Nadam optimizers.
  * Dropout - parameter choice has high impact, suggested the use of 
  * Number of Hidden Units - parameter choice has low impact, suggested the use of 100 units as a good rule of thumb.
  * Batch size - parameter choice has medium impact, suggested 8-32 for large training set.
* In parallel with consulting papers, we also performed Hyperparameter Tuning to get a better of understanding of possible parameter configurations:
  * Due to limited time and computational resources, we decided to tune our model on the following parameters:
    * Number of hidden units per LSTM layer = {1, 2, 3}
    * Number of LSTM layers = {32, 64, 128, 256, 512}
    * Dropout Rate per LSTM layer = {0.1, 0.2, 0.3, 0.4, 0.5}
    * Batch Size = {32, 64, 128}
  * The following are common methods used in hyperparameter tuning:
    * Grid Search - Doing a brute force search across all possible configurations (time consuming).
    * Random Search - Randomly sample from the search space for possible model configuration.
    * Bayesian Optimization - Builds a probability model for the model's metric as objective function and use it to select the most promising parameters. 
  * Note: We chose the model's output validation accuracy as the objective function to optimize.
  * We used the Bayersian Optimization approach, particularly using Tree of Parzen estimators (TPE).
  * We used the [HyperOpt](https://github.com/hyperopt/hyperopt) python library that supports optimization algorithms for hyperparameter tuning.
  * Additionally, we explored the usage of [Apache Spark](https://spark.apache.org/docs/3.0.1/) to perform hyperparameter optimizations in parallel for increase in speed.
* Using a combination of these two approaches and our intuition, we came up with a refined model that we used to retrain our model using the previous dataset.
  
**Evaluation Phase**

<div class="center-align">
  <img width="400px", height="auto", src="https://scikit-learn.org/stable/_images/grid_search_cross_validation.png">
  <p> Figure: K-Fold Cross Validation performed on a dataset, splitting it into K parts (Cross-validation: evaluating estimator performance, 2019). </p>
</div>

* We now have a current refined model, that uses optimized parameters using hyperparameter tuning.
* However, our refined model was only trained on a single configuration of splitting the test/training dataset.
* To evaluate our model's generalised performance, we perform K-Fold Cross Validation  
  * K-Fold Cross Validation performs an evaluation of your model based on different possibilities of splitting your test/training dataset.
  * This gives us an average performance of our model after it trains on different parts of our training set.
  * In K-Fold Cross Validation:
    1. Given dataset is split into K Parts, in our case we chose K = 10.
    2. We then trained our refined model on K-1 parts of the dataset.
    3. We then evaluate it's performance on the K'th part of that dataset.
    4. Repeat steps 2-3 for all K parts.
* From K-Fold cross validation, we obtained statistics of our model's performance across each K'th split.
* Finally, we chose the model that performed the best, and run it using our final collected test-data set for final evaluation.
* The above test dataset is a collection of videos that the model has not seen before to create an unbiased test.



