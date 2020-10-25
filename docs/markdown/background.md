### Background

The core of our gesture recognition model is based on using Recurrent Neural Networks (RNN) for sequential data processing of key-points. In this section, we will introduce Recurrent Neural Networks and explain it's inner workings enabling it to learn sequential dependencies in sequence data for model inferencing. We will then introduce an RNN variant, Long Short Term Memory Networks - designed to tackle problems in vanilla Recurrent Neural Networks. Our model heaviliy relies on LSTMs in handling sequence of key-points for gesture recognition. Feel free to skip this section if you are comfortable with how RNNs and LSTMs work.

#### Recurrent Neural Networks

##### Context
* We as humans, rely on sequences to interpret meaning:
  * Example: Given the sentence "I am living in England, I often speak ___ ".
  * Here, we make connections in our brain connecting past words as "context" to make a guess for that empty word - "English".
* We see that in order to understand sequences - the key elements in doing so would be to:
  * Track long-term dependencies between data points in a sequence (using the word "England" as context for sentence guess).
  * Maintain information about data order in a sequence (knowing that a noun must occur based on past three words).
  * Having a 'shared' mechanism where new data can learn using both current and old data.
* Traditional feed-forward neural networks take in a fixed length input and map it to a fixed length output.
* This makes it difficult to train the model to use information from past data points given a sequence as it does not have an internal mechanism to share past data.

##### What are Recurrent Neural Networks?
* Alternatively, the **Recurrent Neural Networks (RNN)** is a neural network architecture - descendent of the typical feed foward neural network.
* The internal structure uses a recurrence mechanism, where the model accepts both the input data and last hidden state's output when doing prediction or classification.  

<div class="row center-align">
    <div class="col s12 m12 l6">
        <img src="images/Model/FF_NN.png">
        <p> Figure: Feed-forward neural network architecture, mapping fixed length input to fixed length output. </p> 
    </div>
    <div class="col s12 m12 l6">
        <img src="images/Model/RNN_MAIN.png">
        <p> Figure: Recurrent Neural Network architecture, where output is a function of both input and past hidden states. </p> 
    </div>
</div>

##### Types of RNNs
<br>
<div class="center-align">
    <img width="650px;", height="auto" src="images/Model/rnn_types.jpg">
    <p> Figure: Variations of Recurrent Neural Network types (Stanford Recurrent Neural Networks, 2017) </p>
</div>

* There are four variants for RNN types:
  * One-to-many : Given a single input - generate an output sequence (eg: Generate an image caption given a single image)
  * Many-to-one : Given an input sequence - generate a single output (eg: Sentiment Classification, Time Series Classification)
  * Many-to-many [1]: Given an input sequence of length, n - generate an output sequence of different length (eg: Machine Translation from English to Dutch)
  * Many-to-many [2]: Given an input sequence of length, n - generate an output sequence of length, n (eg: Name-Entity recognition)

* As you can see, our problem relates to the "Many-to-one" scenario where:
  * Given 75 keypoint arrays as a sequence.
  * Generate a resulting "classification" of what that sign is.

##### How do RNNs work?
Here, we will briefly discuss details on how RNNs share states from the past when performing inferencing on sequence data.
* To use an RNN model, an input sequence must first be provided with some length N. 

<div class="center-align">
  <img width="500px", height="auto", src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png">
  <p> Figure: Recurrent Neural Networks in both rolled (left) and unrolled illustrations (right) (Colah, 2015) </p>
</div>

* We can see that from figure above, the unrolled version of the RNN model can be seen as a chain of recurrences of itself on the right.
* For all time steps (except the first one), the output h(t) is a function of both the current data point x(t) and the cell's last output h(t-1) creating output h(t). 

* To mathematically represent this, we have:
\\[h_{t} = f_{W}(h_{t-1}, x_{t}) \\]
* where
  * \\( h_{t} \\) cell output,
  * \\( f_{W} \\) function of the RNN cell, parametrized by weight W,
  * \\( h_{t-1} \\) cell's last output, 
  * \\( x_{t} \\) input data point from sequence,

<div class="center-align">
  <img width="500px", height="auto", src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png">
  <p> Figure: Zooming into the RNN recurrence chain (Colah, 2015) </p>
</div>

* Zooming into each RNN cell (as seen in figure above), we see that the following operations occurs:
  * A weight matrix is applied to the cell's last output \\(W_{h}h_{t-1}\\)
  * A weight matrix is also applied to the current input \\(W_{x}x_{t}\\)
  * Note: This weight matrix is repeatedly use for all timesteps by the cell.
  * Both are then concatenated together (stacking vectors) \\(W_{h}h_{t-1} + W_{x}x_{t}\\)
  * Then an activation function (tanh) is applied \\( h_{t} = tanh(W_{h}h_{t-1} + W_{x}x_{t}) \\)
  * The tanh function squashes this output into a range between (-1, 1).
  * Output now set as past cell output for the next point in time t+1, and process repeats for all N data points.

##### Problem with Vanilla RNNs
<div class="center-align">
  <img width="800px", height="auto", src="images/Model/bprop_time.PNG">
  <p> Figure: Gradient flow when performing backpropogation through time for RNNs (Fei Fei Li et al, 2017) </p>
</div>

* Recurrent Neural Networks are good at using past data in a sequence in prediction.
* However, if given a very long sequence (large N), RNNs suffer to keep track of long term dependencies.
* This is due to the **Vanishing Gradient Problem** where
  * After a loss is computed in training, the error is backpropogated through time for weight adjustment.
  * However, backpropogation through time involves a chain of weights being repeatedly multipled when performing gradient descent.
  * Recall that when a real 0 < number < 1 is multiplied by itself many times, it vanishes to zero.
  * We see this problem occurs when the weights try to adjust using inputs way past in the sequence. 
* To circumvent this issue, a new variation of the RNN was proposed called Long Short Term Memory (LSTM) Networks.

##### Long Short Term Memory Networks (LSTM)

<div class="center-align">
  <img width="500px", height="auto", src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png">
  <p> Figure: Core internal structure of an LSTM cell (Colah, 2015). </p>
</div>

* A variation based off from generic RNNs.
* Avoids the vanishing gradient problem by choosing what to remember and what to forget in the data sequence.
* In principle, LSTMs work the same way as RNNs do, except for a difference in each internal cell structure.
* In a single LSTM cell, 
  * A new parameter is also involved, \\(c_{t} \\), known as the cell state.
  * Cell State \\(c_{t}\\) - is a vector that stores the "global" information of sequence data.
  (can be thought of as a conveyor belt of information for the cell to use/update/forget) 
  <div class="center-align">
    <img width="500px", height="auto", src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png">
    <p> Figure: Cell State of an LSTM cell (Colah, 2015) </p>
  </div>
  * Unlike the RNN, the output of the LSTM is a function of three parameters: \\[h_{t} = f_{W}(h_{t-1}, x_{t}, c_{t}) \\]
* To control specific updates in the cell, it uses these internal gates:
  * Forget Gate
    <div class="center-align">
      <img width="300px", height="auto", src="images/Model/forget_gate_real.png">
      <p> Figure: Forget Gate section of the LSTM cell (Colah, 2015) </p>
    </div>

    * Goal: controls whether to forget/erase the current cell state.
      \\[f = sigmoid(W_{h}h_{t-1} + W_{x}x_{t}) \\]
    * Both past cell output \\(h_{t-1}\\) and current input \\(x_{t}\\) are stacked together and passed into a sigmoid function.
    * This outputs \\(f\\) a value between 0 to 1, acting as a "gate" to turn the cell state on or off for this cell (used later in the input gate stage). 
  
  * Input Gate
    <div class="row center-align">
      <div class="col s12 m12 l6">
      <img width="300px", height="220px", src="images/Model/forget_gate.png">
      <p> Figure: First half of operation, creating i, input gate value and C, a potential candidate to be added into current cell state (Colah, 2015) </p>
      </div>
      <div class="col s12 m12 l6">
      <img width="300px", height="220px", src="images/Model/input_gate.png">
      <p> Figure: Second half of the operation, showing how cell state is updated using c,i and f (from forget gate) (Colah, 2015) </p>
      </div>
    </div>

    * Goal: controls update of current cell state using current inputs and past cell state values.
      \\[input = sigmoid(W_{h}h_{t-1} + W_{x}x_{t}) \\]
      \\[candidate = tanh(W_{h}h_{t-1} + W_{x}x_{t}) \\]
      \\[C_{t} = (f) * (c_{t-1}) + (input)*(candidate) \\]
    * Both past cell output \\(h_{t-1}\\) and current input \\(x_{t}\\) are stacked together and passed into a sigmoid function to get the \\(input\\).
    * The \\(candidate\\) is computes the potential value to be added to the current cell state \\(c_{t}\\)
    * The new cell state \\(C_{t}\\) is then updated as the sum of 
      * \\((f) * (c_{t-1})\\) - choosing whether to forget/remember past cell state.
      * \\(input*candidate\\) - choosing whether to input new potential candidate to the current cell state.
  * Output Gate
      <div class="center-align">
        <img width="300px", height="auto", src="images/Model/output_gate.png">
        <p> Figure: Output gate, that controls  (Colah, 2015) </p>
      </div>
      
      * Goal: controls the cell output using current inputs and the cell state.
      \\[o_{t} = sigmoid(W_{h}h_{t-1} + W_{x}x_{t}) \\]
      \\[h_{t} = o_{t}*tanh(C_{t}) \\]
      * Here, the output \\(o_{t}\\) is the sigmoid activation of the inputs (last hidden layer output and current input).
      * Then the final output of this cell is computed as the an element wise multiplication of output \\(o_{t}\\) and cell state \\(tanh(C_{t})\\)
* This recurrence repeats for each time step (just like in the RNN case).

