import random
from random import randint
import time
import os
import tracemalloc

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import backend as keras_backend
import tensorflow as tf



#---------------------------------------------------------------------------------------------
# issue on memory leak;
# src - https://stackoverflow.com/questions/42886049/keras-tensorflow-cpu-training-sequential-models-in-loop-eats-memory?rq=1
# src - https://stackoverflow.com/questions/55742356/training-keras-models-in-a-loop-tensor-is-not-an-element-of-this-graph-when-s
#---------------------------------------------------------------------------------------------


# global constants
n_steps = 75 
n_hidden = 34 # Hidden layer num of features
n_classes = 5
batch_size = 64

#---------------------------------------------------------------------------------------------
# auxiliary tools to load the networks inputs
# definitions:
# 1. X is the raw data;
# 2. Y is the label
#---------------------------------------------------------------------------------------------

def load_X(X_path):
	'''
	args - file path sfor the X-data;
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

def load_Y(y_path):
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
# (stratified) k-fold cross-validation (CV);
# 1. we use stratified version of CV as we have imbalanced dataset; (i.e. not all classes are uniform);
# 2. k-fold = 10, by standard practice of applied machine learning;
# src - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
#  - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/#:~:text=Keras%20can%20separate%20a%20portion,size%20of%20your%20training%20dataset.
#  - https://scikit-learn.org/stable/modules/cross_validation.html
#  - https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#:~:text=Nested%20cross%2Dvalidation%20(CV),its%20(hyper)parameter%20search.&text=Information%20may%20thus%20%E2%80%9Cleak%E2%80%9D%20into,model%20and%20overfit%20the%20data.
#---------------------------------------------------------------------------------------------

# added an extra argument to use different models;
def cross_validate(x_raw, y_raw, kfold, LSTM_func):

	# trace the memory usage;
	tracemalloc.start()

	# load the raw data;
	x_train = load_X(x_raw)
	y_train = load_Y(y_raw)
	
	# load the relevant hyperparameters;
	par = super_params()
	
	#---------------------------------------------------------------
	# define a stratified kfold cross validation test harness;
	# turn on the random shuffling since our data has an order;
	#---------------------------------------------------------------
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
	csv_scores = []
	track = 0
	# now, execute the cross-validation;
	for train_index, test_index in skf.split(x_train, y_train):

		snapshot = tracemalloc.take_snapshot()

		# set up the lstm ;
		#model = LSTM_setup(x_train, y_train)
		model = LSTM_func(x_train, y_train)
		
		# Optimizer
		opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
		
		# Compile model
		model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
		
		# fit the model using the training set;
		model.fit(x_train[train_index], y_train[train_index], epochs=par['epoch'], batch_size = par['batch_size'], verbose = 0)
		
		# resetting the state after every model evaluation;
		# otherwise, the global state maintained by tensorflow might overload;
		# src - https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
		keras_backend.clear_session()
		
		# evaluate the model using the validation set and store each metric;
		scores = model.evaluate(x_train[test_index], y_train[test_index], verbose=0)
		print("fold: %d;    %s: %.2f%%" % (track, model.metrics_names[1], scores[1]*100))
		csv_scores.append(scores[1] * 100)
		track += 1
	
	# now average the metrics across the evaluations and display the results;
	print("averaging: %.2f%% (+/- %.2f%%)" % (np.mean(csv_scores), np.std(csv_scores)))
	
	# what is the statistics on the memory usage?
	top_stats = tracemalloc.take_snapshot().compare_to(snapshot, 'lineno')
	print("statistics on the memory usage (top 5);")
	for stat in top_stats[:5]:
			print(stat)

#---------------------------------------------------------------------------------------------
# extract the best model in terms of accuracy for deployment;
# note:
#   1. since the model is of stochastic nature, so for a fixed set of hyperparameters and architecture,
#       we will different model every time we train and fit it;
#   2. as such, we ought to run for multiple times, and get the best for deployment;
# remark;
#   1. this stage is after we have optimize/tuning the hyperparameters;
#   2. i.e. once we have done: validation, tuning, validation (again), and testing; 
#---------------------------------------------------------------------------------------------

def get_deployable_model(x_raw, y_raw):
	# load the raw data;
	x_train = load_X(x_raw)
	y_train = load_Y(y_raw)
	# set up the model;  
	par = super_params()
	model = LSTM_setup(x_train, y_train)
	# iterating ....

	# define checkpoints to get the best model locally during training;


# test driver;
if __name__ == '__main__':
	xtrain_path = "./training_files/X_train.txt"
	ytrain_path = "./training_files/Y_train.txt"
	path = "C:\\Users\\yongw4\\Desktop\\AUSLAN-DATABASE-YES\\speed_08\\AMBULANCE\\Y_ambulance_train.txt"
	Y_train = load_Y(path)
	print(Y_train)
	print(Y_train.shape)
	
	'''
   # test - 01
   # the auxiiliary functions to load up the txt files;
	X_train = load_X(xtrain_path)
	Y_train = load_Y(ytrain_path)
	print(Y_train.shape)
	print(X_train.shape)
	'''
	# test - 02
	# cross-validation script;
	'''
	kfold = 3
	cross_validate(xtrain_path, ytrain_path, kfold)
	'''

		