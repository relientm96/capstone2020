### Model Development

#### Problem Formulation
* From the Human Pose Estimation & Dataset Processing sections, we've covered the methods, processes and tools used when doing **feature extraction** in order to train our model to classify signs.
* Recall that we formulate the problem of sign language recognition as time-series classification where:
  * Given a series of sequential N data points in time
  * Train a model to classify specific class for that given sequence of N data points.
* Similarly, looking into sign language recognition in our case:
  * We are given a series of sequential 75 key-point arrays (stored as X.txt and Y.txt)
  * Train the model to **classify** that particular sign.

#### Feed Forward Neural Networks (Vanilla Neural Networks)
* Before understanding time series classification in AI models, we would briefly go through an explanation of how feed forward neural networks work.
* Feed-forward neural networks are the most commonly seen architectures.
* It is based off a **single-layer perceptron**:
  * Given N number of X inputs (x1, x2, ... xN)
  * Each input is given a weight (w1, w2, ... wN)
  * It then sums up all N number of x inputs together weighted with respective weights 
  * sum = x1w1 + x2w2 + ... xNwN
  * This sum is then passed on to an "activation function" (eg: sigmoid, tanh, relu)
  * Thus the mathematical representation of a single layer perceptron is:
  * y = activation( x1w1 + x2w2 + ... xNwN)


<div class="center-align">
  <img width=500px height=auto src="https://www.tutorialspoint.com/tensorflow/images/single_layer_perceptron.jpg">
  <p> Figure : Sample diagram of a single layer perceptron. </p>
</div>

* The Feed forward neural network (also known as the Multi-Layer Perceptron) is just a densely connected network, consisting of interconnections between neurons that are single-layer perceptrons.

<div class="center-align">
  <img width=500px height=auto src="https://www.researchgate.net/profile/Mohamed_Zahran6/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png">
  <p> Figure : Sample diagram of a feed-forward neural network - consisting of a network of single-layer perceptrons</p>
</div>

* [At a high level] - To train the model, 
  * It requires training data - having inputs and outputs labelled.
  * All weights in each layer are initialized randomly before training begins.
  * Input is passed into the first layer of the neural network.
  * The results are then "forward propagated" meaning that outputs from each layer are propagated to the next where each neuron performs operations described in the single-layer perceptron model.
  * At the output, it computes the error of it's initial guess with respect to correctly labelled output using a loss function.
  * The error is then "back propogated" allowing the model to re-adjust each layer's weights - called Gradient Descent.
  * Processes repeats until error from loss function is minimised.
  * The trained model is then the resulting tuned layer's weights after training.

#### Recurrent Neural Networks
* We as humans, rely on sequences to interpret meaning:
  * Example: Given the sentence "I am ".
  * Here, we make connections in our brain connecting past words as "context" to make a guess for that empty word - "English".
* We see that in order to understand sequences - the key elements in doing so would be to:
  * Track long-term dependencies between data points in a sequence.
  * Maintain information about data order in a sequence.
  * Having a 'shared' mechanism where new data can learn using both current and old data.
* As we've seen previously, the typical feed-forward neural network is limited in handling all of these problems.
* Alternatively, the **Recurrent Neural Networks (RNN)** is a different neural network architecture - descendent of the typical feed foward neural network.
* At a high level, RNNs have:
  * Input data as a sequence of data points - each point having a time variable to indicate it's position in the sequence.
  * At each time point:
    * x_t = input at time t,
    * h_t =  
    * y_t = output of model 
<div class="center-align">
    <img width="500px;", height="auto" src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
    <p> Figure: High level view of Recurrent Neural Networks (Understanding LSTM Networks, 2015) </p>
</div>

<br>

<div class="center-align">
    <img width="650px;", height="auto" src="images/Model/rnn_types.jpg">
    <p> Figure: Variations of Recurrent Neural Network types (Stanford Recurrent Neural Networks, 2017) </p>
</div>

* A one-to-one correspondence refers to how a typical feed-forward neural network would function, having an input (red) passing into the model (green) to produce an output (blue).
* For RNNs, there exist four different types:
  * One-to-many : Given a single input - generate an output sequence (eg: Generate an image caption given a single image)
  * Many-to-one : Given an input sequence - generate a single output (eg: Sentiment Classification, Time Series Classification)
  * Many-to-many: Given an input sequence - generate an output sequence (eg: Machine Translation - translating Chinese to English)
* As you can see, our problem relates to the "Many-to-one" scenario where:
  * Given 75 keypoint arrays as a sequence.
  * Generate a resulting "classification" of what that sign is.

##### Main Problem with Generic RNNs
* RNNs are great at understanding time dependencies for sequential data.
* However, the issue with RNNs is that they are likely affected by the **Vanishing Gradient Problem**.
* To circumvent this issue, a new variation of the RNN was proposed called Long Short Term Memory (LSTM) Networks.

#### Long Short Term Memory Networks (LSTM)
* A variation based off from generic RNNs.
* Capable of solving the Vanishing Gradient Problem.
[Work in progress]
  



