#!/usr/bin/env python
# source - https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
# author - stuarteiffert
# version - draft;
# modified/abstracted/commented by - nebula M78 team; capstone2020

# useful tricks
# to suppress output of each cell; https://stackoverflow.com/questions/23692950/how-do-you-suppress-output-in-ipython-notebook

#---------------------------------------------------------------------------------------------
# importing the built in packages
#---------------------------------------------------------------------------------------------
# note: version==1.13.1; # this py file doesnt work for version > 2 
# need to refactor for tf ver > 2 migration eventually;
import tensorflow as tf 

# suggested wraparound but not a complete solution;
#import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

# from IPython import get_ipython # a failed wraparound....
import warnings
# reference - https://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython/9031848
warnings.filterwarnings('ignore') # suppress;
#warnings.filterwarnings(action='once') # display the warnings once;

# to be abstracted in a subfile;

import numpy as np
import matplotlib # not in the environment previously
import matplotlib.pyplot as plt
from sklearn import metrics # not in the environment previously
import random
from random import randint
import time
import os

#---------------------------------------------------------------------------------------------
# importing tools from py files;
#---------------------------------------------------------------------------------------------
# import LSTM_utilities as lstm # doesnt work for now due to NoModuleFound Error!;
# so shall include the tools here directly; reminder- to solve the error;

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
	blocks = int(len(X_) / n_steps)
	
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

def LSTM_RNN(_X, _weights, _biases):
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
	lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
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
		index = random.choice(_unsampled)
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

# check for gpu access?
tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(tf.VERSION)


#---------------------------------------------------------------------------------------------
# Preparing dataset:
#---------------------------------------------------------------------------------------------

# Output classes to learn how to classify
LABELS = [    
	"JUMPING",
	"JUMPING_JACKS",
	"BOXING",
	"WAVING_2HANDS",
	"WAVING_1HAND",
	"CLAPPING_HANDS"
] 

#**********************************************************
# REMINDER - change the reference for your local workspace;
#**********************************************************

# DATASET_PATH = "dataset/"
DATASET_PATH = "C:\\Users\\yongw4\\Desktop\\yick\\lstm-tutorial\\dataset\\"
X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

# "rolling window"
n_steps = 32 # 32 timesteps per series

#---------------------------------------------------------------------------------------------
# Load the networks inputs
#---------------------------------------------------------------------------------------------
#X_train = lstm.load_X(X_train_path)
#X_test = lstm.load_X(X_test_path)

#y_train = lstm.load_y(y_train_path)
#y_test = lstm.load_y(y_test_path)

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

#---------------------------------------------------------------------------------------------
# Set the (Hyper)Parameters:
#---------------------------------------------------------------------------------------------

# Input Data 
training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])  # num input parameters per timestep

n_hidden = 34 # Hidden layer num of features
n_classes = 6 

#updated for learning-rate decay
# calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
decaying_learning_rate = True
learning_rate = 0.0025 #used if decaying_learning_rate set to False
init_learning_rate = 0.005
decay_rate = 0.96 #the base of the exponential in the decay
decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

global_step = tf.Variable(0, trainable=False)
lambda_loss_amount = 0.0015

training_iters = training_data_count *300  # Loop 300 times on the dataset, ie 300 epochs
batch_size = 512
display_iter = batch_size*8  # To show test set accuracy during training

print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("\nThe dataset has not been preprocessed, is not normalised etc\n")
print('hello\n')

#---------------------------------------------------------------------------------------------
# to build the neural network;
#---------------------------------------------------------------------------------------------

# Graph input/output
# warning - the attribute: "placeholder" is deprecated for tensorflow versions > 2
# https://better-coding.com/solved-attributeerror-module-tensorflow-has-no-attribute-placeholder/

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
	'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
	'hidden': tf.Variable(tf.random_normal([n_hidden])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

#pred = lstm.LSTM_RNN(x, weights, biases)
pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
	tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
if decaying_learning_rate:
	learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)


#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#---------------------------------------------------------------------------------------------
# Train the network:
#---------------------------------------------------------------------------------------------

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

#________________________________________________________________
# create saver object before training;
# this is needed for saving and restoring the model;
# keep a maximum of ?? models;
saver = tf.train.Saver(max_to_keep=7)
RETRAIN = True
#_________________________________________________________________


# Perform Training steps with "batch_size" amount of data at each loop. 
# Elements of each batch are chosen randomly, without replacement, from X_train, 
# restarting when remaining datapoints < batch_size
step = 1
time_start = time.time()
unsampled_indices = range(0,len(X_train))
# svae the model during training after every "save_step" iterations;
save_step = int(training_iters/3)
		

# temporarily assigned for testing/debugging;
# training_iters = 100
print('batch_size\n\n' , batch_size)
print('training iters\n\n', training_iters)

if RETRAIN:		
	TRAINING_STEPS = step*batch_size
	while TRAINING_STEPS <= training_iters:
		#print (sess.run(learning_rate)) #decaying learning rate
		#print (sess.run(global_step)) # global number of iterations
		if len(unsampled_indices) < batch_size:
			unsampled_indices = range(0,len(X_train))
	
		# batch_xs, raw_labels, unsampled_indicies = lstm.extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
		batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
	
		#batch_ys = lstm.one_hot(raw_labels)
		batch_ys = one_hot(raw_labels)

		# check that encoded output is same length as num_classes, if not, pad it 
		if len(batch_ys[0]) < n_classes:
			temp_ys = np.zeros((batch_size, n_classes))
			temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
			batch_ys = temp_ys

		# Fit training using batch data
		_, loss, acc = sess.run(
			[optimizer, cost, accuracy],
			feed_dict={
				x: batch_xs, 
				y: batch_ys
			}
		)
		train_losses.append(loss)
		train_accuracies.append(acc)
	
		# Evaluate network only at some steps for faster training: 
		if (TRAINING_STEPS % display_iter == 0) or (step == 1) or (TRAINING_STEPS > training_iters):
		
			# To not spam console, show training accuracy/loss in this "if"
			print("Iter #" + str(TRAINING_STEPS) +               ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) +               ":   Batch Loss = " + "{:.6f}".format(loss) +               ", Accuracy = {}".format(acc))
		
			# Evaluation on the test set (no learning made here - just evaluation for diagnosis)
			loss, acc = sess.run(
				[cost, accuracy], 
				feed_dict={
					x: X_test,
					# y: lstm.one_hot(y_test)
					y: one_hot(y_test)
				}
			)
			test_losses.append(loss)
			test_accuracies.append(acc)
			print("PERFORMANCE ON TEST SET:             " +               "Batch Loss = {}".format(loss) +               ", Accuracy = {}".format(acc))
		
		# safeguarding; back up the model during training after every fixed iterations;
		# reference: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		# saver.save(sess, 'iter_model', global_step = step, write_meta_graph = False); # since the graph is fixed throughout;
		if (TRAINING_STEPS % (100*batch_size)) == 0:
            # [issue] right now save at the current directory; failed at relative path; 
			save_path = saver.save(sess, "iter_model", global_step = step)
			print("Model saved in file: %s" % save_path)
		
		# increment;
		step += 1
		TRAINING_STEPS = step*batch_size
		
	print("Optimization Finished!")

#check if you want to retrain or import a saved model
else:
	print('restoring the saved trained model\n')
	# reference: https://stackabuse.com/tensorflow-save-and-restore-models/
	tf.reset_default_graph()
	imported_meta = tf.train.import_meta_graph("saved_final_model.meta")
	imported_meta.restore(sess, tf.train.latest_checkpoint('./'))
	print("Model restored.")

#_____________________________________________________________________
# Check if you want to save your current model after training;
if RETRAIN:
	save_path = saver.save(sess, "saved_final_model")
	print("Model saved in file: %s" % save_path)
#____________________________________________________________________

#---------------------------------------------------------------------------------------------
# Accuracy for test data
#---------------------------------------------------------------------------------------------
one_hot_predictions, accuracy, final_loss = sess.run(
	[pred, accuracy, cost],
	feed_dict={
		x: X_test,
		#y: lstm.one_hot(y_test)
		y:one_hot(y_test)
	}
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print("FINAL RESULT: " +       "Batch Loss = {}".format(final_loss) +       ", Accuracy = {}".format(accuracy))
time_stop = time.time()
print("TOTAL TIME:  {}".format(time_stop - time_start))


#-----------------------------------------------------------------------
# offline prediction;
#-----------------------------------------------------------------------

X_val_path = DATASET_PATH + "X_val.txt"
X_val = load_X(X_val_path)
print(X_val)

preds = sess.run(
	[pred],
	feed_dict={
		x: X_val
   }
)
print('offline evaluation\n')
print(preds)

#sess.close()
print('\n test accuracies \n')
print(test_accuracies)