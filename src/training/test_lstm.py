#/usr/bin/env/python
# created by matthew; nebulaM78 team; capstone 2020;
# to train the (cudnn) lstm model 

# tf/ keras
import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# known issue: keras version mismatch;
# solution src - https://stackoverflow.com/questions/53183865/unknown-initializer-glorotuniform-when-loading-keras-model
#from keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History 
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# sklearn tool kits
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# opthers;
import numpy as np
from datetime import datetime
import tracemalloc
from contextlib import redirect_stdout
import os
import sys
import json
try:
	import cPickle as pickle
except ImportError:  # python 3.x
	import pickle

# own modules;
import LSTM_tools as lstm_tools
import file_tools as file_tools
import LSTM_model as MODEL

# check for gpu cuda access;
lstm_tools.check_gpu()

#----------------------------------------------------------------------
# path list;
#----------------------------------------------------------------------
prefix = "C:\\Users\\yongw4\\Desktop\\train-21-10-2020\\train-21-10-2020"
noframes = "\\models\\35-frames"
nframes = str(35)
tmpname = prefix + noframes 
# final saved model, if there's no early stopping;
filepath = tmpname + "\\fmodel.h5"

#----------------------------------------------------------------------
# constants; 
# 1. the hyperparameters;
#----------------------------------------------------------------------
n_hidden = 128 # hidden layer number of features;
n_classes = 4  # number of sign classes;
batch_size = 64

#----------------------------------------------------------------------
# LOADING THE DATA:
# - X.txt; the keypoints;
# - Y.txt; the corresponding labels;
#----------------------------------------------------------------------
tmpname = prefix +  "\\train-npy\\35-frames"
np_X = tmpname+  "\\X_MAIN.npy"
np_Y = tmpname+  "\\Y_MAIN.npy"

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
def cross_validate(x_raw, y_raw, kfold, LSTM_func,  par, log_path):

	# trace the memory usage;
	tracemalloc.start()

	# load the raw data;
	x_train = np.load(x_raw)
	y_train = np.load(y_raw)
	
	# load the relevant hyperparameters;
	#par = MODEL.super_params()
	
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
		model = LSTM_func(x_train, y_train, par)
		
		# Optimizer
		opt = tf.keras.optimizers.Adam(lr=1e-4, decay=1e-5)
		model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])

		# fit the model using the training set;
		earlyStopping = EarlyStopping(monitor='val_acc',patience=30,verbose=1,mode='max')
		callbacks_list = [earlyStopping]
		model.fit(x_train[train_index], y_train[train_index], epochs=500, batch_size = 64, verbose = 1, callbacks=callback_list)
		
		# evaluate the model using the validation set and store each metric;
		scores = model.evaluate(x_train[test_index], y_train[test_index], verbose=0)
		print("fold: %d;    %s: %.2f%%" % (track, model.metrics_names[1], scores[1]*100))
		csv_scores.append(scores[1] * 100)
		track += 1
	
	# now average the metrics across the evaluations and display the results;
	print("csv_score list: ", csv_scores)
	print("averaging: %.2f%% (+/- %.2f%%)" % (np.mean(csv_scores), np.std(csv_scores)))
	
	# what is the statistics on the memory usage?
	top_stats = tracemalloc.take_snapshot().compare_to(snapshot, 'lineno')
	print("statistics on the memory usage (top 5);")
	for stat in top_stats[:5]:
			print(stat)

	# logging the model stats;
	with open(log_path, 'a+') as f:
		with redirect_stdout(f):
			model.summary()

	with open(log_path, 'a+') as f:
		line1 = "compensate imbalance class distribution method: stratification"
		f.write("\n{}\n".format(line1))

	with open(log_path, "a+") as f:
		line1 = datetime.now().strftime("%Y-%m-%d %H:%M")
		line2 = str(x_train.shape)
		line3 = str(kfold)
		line4 = str(csv_scores)
		line5 = np.mean(csv_scores)
		line6 = np.std(csv_scores)
		f.write("\n{}\n data size: {}\n kfold: {}\n scores: {}\n mean: {}\n std: {}%\n".format(line1, line2, line3, line4, line5, line6))




# ----------------------------------------------------------------------------
# experimentally get the average;
# src - https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/
# ----------------------------------------------------------------------------
def evaluate_model(text_x, test_y, model_path):
	model = load.model(model_path)
	_, accuracy = model.evaluate(test_x, test_y, verbose=0)
	return accuracy

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
	return m,s

def exp_avg(repeats, text_x, test_y, model_path, log_path, classtype):
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(test_x, test_y, model_path)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	
	# summarize results
	mean_info ,std_info = summarize_results(scores)	

	# logging the model stats;
	with open(log_path, "a+") as f:
		line1 = datetime.now().strftime("%Y-%m-%d %H:%M")
		line2 = "classname: " + classtype
		line3 = mean_info
		line4 = std_info
		f.write("\n{}\n {}\n mean: {}\n std: {}\n".format(line1, line2, line3, line4))


def drive_exp_avg(prefix, model_path, log_path):
	#prefix_test = "C:\\Users\\yongw4\\Desktop\\test-set\\test-set\\test-npy\\frame-35\\processed\\classes"

	test_X = []
	test_Y = []
	curate_list = []
	length = 0
	for root, dirs, files in os.walk(prefix_test, topdown=False):
		for name in files:
			tmp0 = name.split("_")
			tmp = tmp0[0].lower()
			classnum = tmp0[-1].split(".")[-2]
			if(length < 4):
				curate_list.append(classnum)
			length = length + 1
			if(tmp == "x"):
				loc = os.path.join(root,name)
				test_X.append(loc)
			elif(tmp == "y"):
				loc = os.path.join(root,name)
				test_Y.append(loc)
			else:
				continue

	#---------------------------------------
	# evaluating the test set;
	#---------------------------------------
	model = load.model(model_path)
	# logging the model stats;
	with open(log_path, 'a+') as f:
		with redirect_stdout(f):
			model.summary()

	for i in range(0, len(test_X)):
		exp_avg(30, text_X[i], test_Y[i], model_path, log_path, curate_list[i])


# test driver;
if __name__ == '__main__':
  

    '''
	#---------------------------------------
	# cross validation
	#---------------------------------------
	
	prefix = "C:\\Users\\yongw4\\Desktop\\train-21-10-2020\\train-21-10-2020\\train-npy\\75-frames"
	X_train = prefix+"\\X_combine.npy"
	Y_train = prefix+"\\Y_combine.npy"
	log_path = prefix + "\\log_cross_validate.txt"
	kfold = 10
	
	MODELS = [MODEL.lstm_tanh_one, MODEL.lstm_tanh_one, MODEL.lstm_tanh_one,  MODEL.lstm_tanh_two, MODEL.lstm_tanh_two, MODEL.lstm_tanh_two]
	par1 = MODEL.super_params(n_hidden=32)
	par2 = MODEL.super_params(n_hidden=64)
	par3 = MODEL.super_params(n_hidden=128)
	par4 = MODEL.super_params(n_hidden=32)
	par5 = MODEL.super_params(n_hidden=64)
	par6 = MODEL.super_params(n_hidden=128)
	PARAMS = [par1, par2, par3, par4, par5, par6]

	for i in range(0, len(MODELS)):
		cross_validate(X_train, Y_train, kfold, MODELS[i], PARAMS[i], log_path)
	'''






