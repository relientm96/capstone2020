#!/usr/bin/env python
# source - https://github.com/stuarteiffert/RNN-for-Human-Activity-Recognition-using-2D-Pose-Input
# author - stuarteiffert
# modified/abstracted (on-hold) /commented by - yick


# useful tricks
# to suppress output of each cell; https://stackoverflow.com/questions/23692950/how-do-you-suppress-output-in-ipython-notebook

#---------------------------------------------------------------------------------------------
# importing the built in packages
#---------------------------------------------------------------------------------------------
# note: version==1.13.1; # this py file doesnt work for version > 2 
# need to refactor for tf ver > 2 migration eventually;
import tensorflow as tf 

import warnings
# reference - https://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython/9031848
warnings.filterwarnings('ignore') # suppress the warning;s 
#warnings.filterwarnings(action='once') # display the warnings once;


# suggested wraparound but not a complete solution;
# import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# yet, another failed wraparound;
# a failed wraparound....
# from IPython import get_ipython 

import numpy as np
import matplotlib # not in the environment previously
import matplotlib.pyplot as plt
from sklearn import metrics # not in the environment previously
import random
from random import randint
import time
import os

#---------------------------------------------------------------------------------------------
# importing self written modules 
#---------------------------------------------------------------------------------------------
import sys
# temporarily add the self-written module path to the environment;
# source - https://stackoverflow.com/questions/32509046/importing-self-written-python-module
sys.path.append("C:\\Users\\yongw4\\Desktop\\yick\\lstm-tutorial")
import LSTM_tools as lstm

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

# "rolling window"; one could view this as a "buffer" for the input;
n_steps = 32 # 32 timesteps per series

#---------------------------------------------------------------------------------------------
# Load the networks inputs
#---------------------------------------------------------------------------------------------
X_train = lstm.load_X(X_train_path)
X_test = lstm.load_X(X_test_path)

y_train = lstm.load_y(y_train_path)
y_test = lstm.load_y(y_test_path)

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

pred = lstm.LSTM_RNN(x, weights, biases, n_input)

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
RETRAIN = False
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
	
		batch_xs, raw_labels, unsampled_indicies = lstm.extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
		batch_ys = lstm.one_hot(raw_labels)
		
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
					y: lstm.one_hot(y_test)
				}
			)
			test_losses.append(loss)
			test_accuracies.append(acc)
			print("PERFORMANCE ON TEST SET:             " +               "Batch Loss = {}".format(loss) +               ", Accuracy = {}".format(acc))
		
		# safeguarding; back up the model during training after every fixed iterations;
		# reference: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
		# saver.save(sess, 'iter_model', global_step = step, write_meta_graph = False); # since the graph is fixed throughout;
		if (TRAINING_STEPS % (1000*batch_size)) == 0:
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
		y: lstm.one_hot(y_test)
	}
)
test_losses.append(final_loss)
test_accuracies.append(accuracy)
print("FINAL RESULT: " +       "Batch Loss = {}".format(final_loss) +       ", Accuracy = {}".format(accuracy))
time_stop = time.time()
print("TOTAL TIME:  {}".format(time_stop - time_start))


# ============================= reverse - engineering; =======================
# to see what form of the predictions returned from "sess.run()
# to investigate how to decode it to correspond to which "activity"
one_hot_predictions = sess.run(
	[pred],
	feed_dict={
		x: X_test,
			}
)
# write it to a file for a clearer view (?);
with open('listfile.txt', 'w') as filehandle:
	for listitem in one_hot_predictions:
		filehandle.write('%s\n' % listitem)

print('type of one_hot_predictions\n', type(one_hot_predictions))
print('one_hot_predictions\n', one_hot_predictions)
print('type of one_hot_predictions[0]\n', type(one_hot_predictions[0]))
print('one_hot_predictions[0]\n', one_hot_predictions[0])
print('decode the one-hot encoder predictions\n', one_hot_predictions[0].argmax(1))

# reversed;
# one_hot_predictions is a list of one multidimensional numpy array;
# ============================================================================

#-----------------------------------------------------------------------
# offline prediction;
#-----------------------------------------------------------------------
print('offline prediction\n')
X_val_path = DATASET_PATH + "X_val.txt"
X_val = lstm.load_X(X_val_path)
print('preview of X_val.txt\n', X_val)

preds = sess.run(
	[pred],
	feed_dict={
		x: X_val
   }
)

# numpy array of the form ONEHOT = [x1,x2,x3,x4,x5,x6]; six floating numbers, x_{i}
offline_onehot_pred = preds[0]  
# offline_onehot_pred.argmax(1) = [int(max(ONEHOT))], a numpy array of one element only;
offline_pred = (offline_onehot_pred.argmax(1))[0]
print('one-hot encoded prediction\n', offline_onehot_pred) 
print("decode the one-hot prediction?\n", offline_pred)

# dictionary to map the decoded index to the corresponding activity (label);
MAP_DICT = {
	0: "JUMPING",
	1: "JUMPING_JACKS",
	2: "BOXING",
	3: "WAVING_2HANDS",
	4: "WAVING_1HAND",
	5: "CLAPPING_HANDS"
}
print("what is the predicted activity?\n", MAP_DICT[offline_pred])

#-----------------------------------------------------------------------
# offline prediction - NOTE;
#-----------------------------------------------------------------------
# 1. X_val.txt has 32 lines;
# 2. pred = sess.run([pred], feed_dict={x: X_val} output a one-hot encoded prediction from the 32-lined input;
# 3. pred = [x0,x1,x2,x3,x4,x5] for x_{i} is a floating number; where i = 0,..,5 correspondes to the activitiy as in MAP_DICT above;
# 4. max{pred} is the predicted activity;
# 5. so, a buffer of {32 reformatted json files} -> one-hot-vector; where each json file corresponds to ONE frame;
# 6. given a time series say with 140 frames; the 140 frames will be divided to 32 each as an input to the buffer to perform the prediction;


# ============================= reverse - engineering; =======================
# to get all of the placeholders in the graph
# source - https://stackoverflow.com/questions/51253483/undestanding-feed-dict-in-sess-run
placeholderss = [ op for op in sess.graph.get_operations() if op.type == "Placeholder"]
print(placeholderss)
# reversed - it has two placeholders as constructed above; i.e.;
# x = tf.placeholder(tf.float32, [None, n_steps, n_input])
# y = tf.placeholder(tf.float32, [None, n_classes])
# ============================================================================

#sess.close()
# print('\n test accuracies \n')
# print(test_accuracies)'''