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
  <p> Figure : Sample diagram of how a single layer perceptron works. </p>
</div>

* The Feed forward neural network (also known as the Multi-Layer Perceptron) is just a densely connected network, consisting of interconnections between neurons that are single-layer perceptrons.

<div class="center-align">
  <img width=500px height=auto src="https://www.researchgate.net/profile/Mohamed_Zahran6/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png">
  <p> Figure : Sample diagram of a feed-forward neural network - consisting of a network of single-layer perceptrons</p>
</div>

* To train the model, it uses labelled input and output data:
  * All weights in each layer are initialized randomly. 
  * Input is passed into the Multi Layer Perceptron
  * The results are then "forward propagated" meaning that outputs from each layer are propagated to the next where each neuron performs operations described in the single-layer perceptron model.
  * At the output, it computes the error of it's guess with respect to the labelled output using a loss function.
  * The error is then "back propogated" allowing the model to re-adjust each layer's weights - called Gradient Descent.
  * The training stops when error from loss function is minmized.
  * The trained model is then the resulting tuned layer's weights after training.

#### Recurrent Neural Networks
* The problem with Feed Forward networks, is that it is hard to train using data for sequence/time series classification.
* We as humans as well, rely on sequences to interpret meaning:
  * Example: Given an incomplete sentence, predict what the next word is in the sentence.
  * Here, we make connections in our brain connecting past words as "context" to make a guess.
* Similarly, the **Recurrent Neural Networks (RNN)** is a different neural network architecture 
* Good at persisting data from previous information given a sequence using a loop.
* We would not go through details of it's working bu highly advice you to look into the explanations of how RNNs work here in [this well written article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
  
<div class="center-align">
    <img width="500px;", height="auto" src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
    <p> Figure: RNN model structure, rolled on the left and unrolled on the right (Understanding LSTM Networks, 2015) </p>
</div>

##### Main Problem with Generic RNNs
* RNNs are great at understanding time dependencies for sequential data.
* However, the issue with RNNs is that they are likely affected by the **Vanishing Gradient Problem**.
* To circumvent this issue, a new variation of the RNN was proposed called Long Short Term Memory (LSTM) Networks.

#### Long Short Term Memory Networks (LSTM)
* A variation based off from generic RNNs.
* Capable of solving the Vanishing Gradient Problem.
[Work in progress]
  



