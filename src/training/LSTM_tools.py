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

#----------------------------------------------------------------------
# check for gpu access;
#----------------------------------------------------------------------
def check_gpu():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if(len(gpus) ==0):
		sys.exit("NO GPU is found!")
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)


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

		