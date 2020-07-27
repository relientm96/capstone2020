import numpy as np
import matplotlib # not in the environment previously
import matplotlib.pyplot as plt
from sklearn import metrics # not in the environment previously
import random
from random import randint
import time
import os
import tensorflow as tf 


# global constants
n_steps = 75 # 32 timesteps per series
n_hidden = 34 # Hidden layer num of features
n_classes = 4

#---------------------------------------------------------------------------------------------
# auxilairy tools to load the networks inputs
# definitions:
# 1. X is the raw data;
# 2. Y is the label
#---------------------------------------------------------------------------------------------

def load_X(X_path):
	'''
	args - file path for the X-data;
	return - formatted X-data content as np.array as the input for the neural network
	'''
	file = open(X_path, 'r')
	X_ = np.array(
		[elem for elem in [
			row.split(',') for row in file
		]], 
		dtype=np.float32
	)
	file.close()
	#print('values\n\n', X_)
	#print('len\n', len(X_))
	#print('len 02\n\n', len(X_[0]))
	#print('values 02\n\n', X_[0])
	
	blocks = int(len(X_) / n_steps)
	#print("hello)")
	X_ = np.array(np.split(X_,blocks))

	return X_ 

# Load the networks outputs

def load_y(y_path):
	'''
	args - file path for the Y-data;
	return - formatted X-data content as np.array as the input for the neural network
	'''
	file = open(y_path, 'r')
	y_ = np.array(
		[elem for elem in [
			row.replace('  ', ' ').strip().split(' ') for row in file
		]], 
		dtype=np.int32
	)
	file.close()
	
	# for 0-based indexing 
	return y_ - 1

#---------------------------------------------------------------------------------------------
# the core functions to build and train the neural network;
#---------------------------------------------------------------------------------------------

def LSTM_RNN(_X, _weights, _biases, n_input):
	# model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

	_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
	_X = tf.reshape(_X, [-1, n_input])   
	# Rectifies Linear Unit activation function used
	_X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
	# Split data because rnn cell needs a list of inputs for the RNN inner loop
	_X = tf.split(_X, n_steps, 0) 

	# Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
	lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	lstm_cell_4 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	
	lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3, lstm_cell_4], state_is_tuple=True)
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

	# A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
	lstm_last_output = outputs[-1]
	
	# Linear activation
	return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, _labels, _unsampled, batch_size):
	# Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
	# Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
	# unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
	
	shape = list(_train.shape)
	shape[0] = batch_size
	batch_s = np.empty(shape)
	batch_labels = np.empty((batch_size,1)) 
	for i in range(batch_size):
		# Loop index
		# index = random sample from _unsampled (indices)
		#print('lstm tools\n',_unsampled)
		index = random.choice(list(_unsampled))
		batch_s[i] = _train[index] 
		batch_labels[i] = _labels[index]
		# yick-modified;
		# reference - https://stackoverflow.com/questions/28150965/why-range0-10-remove1-does-not-work
		#_unsampled.remove(index) # note: '_unsampled' is of class: range; see the reference; 
		_unsampled = [i for i in _unsampled if i != index]
		
	return batch_s, batch_labels, _unsampled


def one_hot(y_):
	# One hot encoding of the network outputs
	# e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
	
	y_ = y_.reshape(len(y_))
	n_values = int(np.max(y_)) + 1
	return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


